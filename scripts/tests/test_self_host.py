from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import tempfile
import unittest
from contextlib import contextmanager, nullcontext, redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import self_host


def completed(
    command: list[str] | tuple[str, ...],
    stdout: str = "",
    *,
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def config() -> self_host.DeploymentConfig:
    return self_host.DeploymentConfig(
        profile="sandbox",
        region="us-west-2",
        account="123456789012",
        partition="aws",
        domain="agent.example.com",
        certificate_arn=(
            "arn:aws:acm:us-west-2:123456789012:"
            "certificate/00000000-0000-0000-0000-000000000000"
        ),
    )


class InputValidationTests(unittest.TestCase):
    def test_domain_normalizes_case_and_trailing_dot(self) -> None:
        self.assertEqual(
            self_host.normalize_domain(" Agent.Example.COM. "), "agent.example.com"
        )

    def test_domain_rejects_urls_paths_and_single_labels(self) -> None:
        for value in (
            "https://agent.example.com",
            "agent.example.com/path",
            "localhost",
            "-agent.example.com",
            "agent..example.com",
        ):
            with self.subTest(value=value), self.assertRaises(self_host.SelfHostError):
                self_host.normalize_domain(value)

    def test_certificate_must_match_selected_identity(self) -> None:
        value = config().certificate_arn
        self.assertEqual(
            self_host.validate_certificate_arn(
                value,
                account="123456789012",
                region="us-west-2",
                partition="aws",
            ),
            value,
        )
        for change in (
            {"account": "999999999999"},
            {"region": "eu-west-1"},
            {"partition": "aws-us-gov"},
        ):
            kwargs = {
                "account": "123456789012",
                "region": "us-west-2",
                "partition": "aws",
            } | change
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(self_host.SelfHostError, "selected AWS account"),
            ):
                self_host.validate_certificate_arn(value, **kwargs)

    def test_slack_payload_requires_all_verified_shapes(self) -> None:
        valid = {
            "SLACK_CLIENT_ID": "123456789.987654321",
            "SLACK_CLIENT_SECRET": "a" * 32,
            "SLACK_SIGNING_SECRET": "b" * 32,
            "SLACK_APP_ID": "A123456789",
        }
        self.assertEqual(self_host.validate_slack_payload(valid), valid)

        invalid = {
            "SLACK_CLIENT_ID": "not-an-id",
            "SLACK_CLIENT_SECRET": "short",
            "SLACK_SIGNING_SECRET": "Z" * 32,
            "SLACK_APP_ID": "B123",
        }
        for key, bad_value in invalid.items():
            payload = valid | {key: bad_value}
            with self.subTest(key=key), self.assertRaises(self_host.SelfHostError):
                self_host.validate_slack_payload(payload)

    def test_bridge_payload_rejects_missing_or_weak_secrets(self) -> None:
        valid = {
            "BRIDGE_OAUTH_STATE_SECRET": "a" * 64,
            "BRIDGE_GATEWAY_JWT_PRIVATE_KEY_PEM": (
                "-----BEGIN PRIVATE KEY-----\nkey\n-----END PRIVATE KEY-----\n"
            ),
            "ADMIN_SECRET": "b" * 48,
        }
        self.assertEqual(self_host.validate_bridge_payload(valid), valid)
        for payload in (
            valid | {"ADMIN_SECRET": "short"},
            valid | {"BRIDGE_OAUTH_STATE_SECRET": "short"},
            valid | {"BRIDGE_GATEWAY_JWT_PRIVATE_KEY_PEM": "not a key"},
            {"ADMIN_SECRET": "b" * 48},
        ):
            with (
                self.subTest(payload=payload),
                self.assertRaises(self_host.SelfHostError),
            ):
                self_host.validate_bridge_payload(payload)


class ManifestTests(unittest.TestCase):
    def test_generated_manifest_rewrites_every_public_slack_url(self) -> None:
        manifest = self_host.generated_slack_manifest("ops.example.com")
        self.assertEqual(
            manifest["oauth_config"]["redirect_urls"],
            ["https://ops.example.com/slack/oauth/callback"],
        )
        settings = manifest["settings"]
        self.assertEqual(
            settings["event_subscriptions"]["request_url"],
            "https://ops.example.com/slack/events",
        )
        self.assertEqual(
            settings["interactivity"]["request_url"],
            "https://ops.example.com/slack/interactions",
        )
        serialized = json.dumps(manifest)
        self.assertNotIn("agent.example.com", serialized)
        self.assertIn("app_mentions:read", serialized)

    def test_manifest_write_is_private_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "manifest.json"
            self_host.write_generated_manifest("ops.example.com", destination)
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
            self.assertEqual(
                json.loads(destination.read_text())["oauth_config"]["redirect_urls"],
                ["https://ops.example.com/slack/oauth/callback"],
            )

            outside = Path(temporary) / "outside"
            outside.write_text("do not replace", encoding="utf-8")
            destination.unlink()
            destination.symlink_to(outside)
            with self.assertRaisesRegex(self_host.SelfHostError, "symlink"):
                self_host.write_generated_manifest("other.example.com", destination)
            self.assertEqual(outside.read_text(encoding="utf-8"), "do not replace")


class SecretTests(unittest.TestCase):
    def test_read_secret_handles_found_missing_and_permission_failure(self) -> None:
        document = {
            "ARN": "arn:aws:secretsmanager:us-west-2:123456789012:secret:example-abc",
            "SecretString": json.dumps({"KEY": "value"}),
        }
        runner = mock.Mock()
        runner.run.return_value = completed([], json.dumps(document))
        found = self_host.read_secret(runner, config(), "example")
        assert found
        self.assertEqual(found.payload, {"KEY": "value"})
        self.assertTrue(found.existed)

        runner.run.return_value = completed(
            [], returncode=254, stderr="ResourceNotFoundException"
        )
        self.assertIsNone(self_host.read_secret(runner, config(), "missing"))

        runner.run.return_value = completed(
            [], returncode=254, stderr="AccessDeniedException"
        )
        with self.assertRaisesRegex(self_host.SelfHostError, "permissions"):
            self_host.read_secret(runner, config(), "forbidden")

    def test_put_secret_uses_private_temp_file_and_removes_it(self) -> None:
        runner = mock.Mock()
        runner.run.return_value = completed([], "{}")
        stored = self_host.SecretDocument(
            arn="arn:aws:secretsmanager:us-west-2:123456789012:secret:example-abc",
            payload={"TOKEN": "secret-value"},
            existed=True,
        )
        with mock.patch.object(self_host, "read_secret", return_value=stored):
            result = self_host.put_secret(
                runner,
                config(),
                "example",
                {"TOKEN": "secret-value"},
                existed=False,
            )

        self.assertEqual(result.arn, stored.arn)
        command = runner.run.call_args_list[0].args[0]
        self.assertIn("create-secret", command)
        self.assertNotIn("secret-value", command)
        file_arg = next(part for part in command if str(part).startswith("file://"))
        self.assertFalse(Path(str(file_arg).removeprefix("file://")).exists())

    def test_existing_slack_secret_is_reused_without_prompt(self) -> None:
        payload = {
            "SLACK_CLIENT_ID": "123456789.987654321",
            "SLACK_CLIENT_SECRET": "a" * 32,
            "SLACK_SIGNING_SECRET": "b" * 32,
            "SLACK_APP_ID": "A123456789",
        }
        existing = self_host.SecretDocument("arn:example", payload, True)
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("builtins.input", side_effect=AssertionError("prompted")),
            mock.patch("getpass.getpass", side_effect=AssertionError("prompted")),
        ):
            self.assertEqual(self_host.collect_slack_payload(existing), payload)


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_existing_runtime_environment_reads_only_managed_optional_values(self) -> None:
        runner = mock.Mock()
        runner.run.side_effect = [
            completed(
                ["aws", "list-agent-runtimes"],
                json.dumps(
                    {
                        "agentRuntimes": [
                            {
                                "agentRuntimeName": "coreAgent_coreAgent",
                                "agentRuntimeId": "coreAgent_coreAgent-abcdefghij",
                                "agentRuntimeVersion": "7",
                            }
                        ]
                    }
                ),
            ),
            completed(
                ["aws", "get-agent-runtime"],
                json.dumps(
                    {
                        "agentRuntimeName": "coreAgent_coreAgent",
                        "agentRuntimeId": "coreAgent_coreAgent-abcdefghij",
                        "agentRuntimeVersion": "7",
                        "environmentVariables": {
                            "AWS_REGION": "us-west-2",
                            "GITHUB_APP_ID": "3350219",
                            "AGENTCORE_MEMORY_ID": "memory-1",
                            "UNMANAGED_VALUE": "do-not-copy",
                        },
                    }
                ),
            ),
        ]

        result = self_host._existing_runtime_environment(runner, config())

        self.assertEqual(
            result,
            {
                "GITHUB_APP_ID": "3350219",
                "AGENTCORE_MEMORY_ID": "memory-1",
            },
        )
        get_command = list(runner.run.call_args_list[1].args[0])
        self.assertIn("--agent-runtime-id", get_command)
        self.assertIn("coreAgent_coreAgent-abcdefghij", get_command)
        self.assertIn("--agent-runtime-version", get_command)
        self.assertIn("7", get_command)

    def test_existing_runtime_environment_allows_first_install(self) -> None:
        runner = mock.Mock()
        runner.run.return_value = completed(
            ["aws", "list-agent-runtimes"], json.dumps({"agentRuntimes": []})
        )

        self.assertEqual(
            self_host._existing_runtime_environment(runner, config()), {}
        )
        self.assertEqual(runner.run.call_count, 1)

    def test_existing_runtime_environment_rejects_ambiguous_runtime(self) -> None:
        runtime = {
            "agentRuntimeName": "coreAgent_coreAgent",
            "agentRuntimeId": "coreAgent_coreAgent-abcdefghij",
            "agentRuntimeVersion": "7",
        }
        runner = mock.Mock()
        runner.run.return_value = completed(
            ["aws", "list-agent-runtimes"],
            json.dumps({"agentRuntimes": [runtime, runtime]}),
        )

        with self.assertRaisesRegex(self_host.SelfHostError, "found 2"):
            self_host._existing_runtime_environment(runner, config())


class OnboardingEnvironmentTests(unittest.TestCase):
    def test_existing_onboarding_environment_reads_only_github_slug(self) -> None:
        cluster_arn = "arn:aws:ecs:us-west-2:123456789012:cluster/agentcore-services"
        task_arn = "arn:aws:ecs:us-west-2:123456789012:task-definition/onboarding:14"
        runner = mock.Mock()
        runner.run.side_effect = [
            completed(
                ["aws", "ecs", "list-clusters"],
                json.dumps({"clusterArns": [cluster_arn]}),
            ),
            completed(
                ["aws", "ecs", "describe-services"],
                json.dumps(
                    {
                        "services": [
                            {
                                "serviceName": "agentcore-onboarding",
                                "taskDefinition": task_arn,
                            }
                        ],
                        "failures": [],
                    }
                ),
            ),
            completed(
                ["aws", "ecs", "describe-task-definition"],
                json.dumps(
                    {
                        "taskDefinition": {
                            "containerDefinitions": [
                                {
                                    "name": "onboarding",
                                    "environment": [
                                        {
                                            "name": "GITHUB_APP_SLUG",
                                            "value": "slinck-ai",
                                        },
                                        {"name": "NODE_ENV", "value": "production"},
                                    ],
                                }
                            ]
                        }
                    }
                ),
            ),
        ]

        result = self_host._existing_onboarding_environment(runner, config())

        self.assertEqual(result, {"GITHUB_APP_SLUG": "slinck-ai"})
        service_command = list(runner.run.call_args_list[1].args[0])
        self.assertIn(cluster_arn, service_command)
        self.assertIn("agentcore-onboarding", service_command)
        task_command = list(runner.run.call_args_list[2].args[0])
        self.assertIn(task_arn, task_command)

    def test_existing_onboarding_environment_allows_first_install(self) -> None:
        runner = mock.Mock()
        runner.run.return_value = completed(
            ["aws", "ecs", "list-clusters"], json.dumps({"clusterArns": []})
        )

        self.assertEqual(
            self_host._existing_onboarding_environment(runner, config()), {}
        )
        self.assertEqual(runner.run.call_count, 1)

    def test_existing_onboarding_environment_rejects_duplicate_slug(self) -> None:
        runner = mock.Mock()
        runner.run.side_effect = [
            completed(
                ["aws", "ecs", "list-clusters"],
                json.dumps(
                    {
                        "clusterArns": [
                            "arn:aws:ecs:us-west-2:123456789012:"
                            "cluster/agentcore-services"
                        ]
                    }
                ),
            ),
            completed(
                ["aws", "ecs", "describe-services"],
                json.dumps(
                    {
                        "services": [{"taskDefinition": "task:14"}],
                        "failures": [],
                    }
                ),
            ),
            completed(
                ["aws", "ecs", "describe-task-definition"],
                json.dumps(
                    {
                        "taskDefinition": {
                            "containerDefinitions": [
                                {
                                    "name": "onboarding",
                                    "environment": [
                                        {"name": "GITHUB_APP_SLUG", "value": "one"},
                                        {"name": "GITHUB_APP_SLUG", "value": "two"},
                                    ],
                                }
                            ]
                        }
                    }
                ),
            ),
        ]

        with self.assertRaisesRegex(self_host.SelfHostError, "duplicate"):
            self_host._existing_onboarding_environment(runner, config())


class DeploymentPlanTests(unittest.TestCase):
    @contextmanager
    def _guided_deploy(
        self,
        *,
        health_result: bool = True,
        skip_health_check: bool = False,
        existing_runtime_environment: dict[str, str] | None = None,
        existing_onboarding_environment: dict[str, str] | None = None,
        operator_environment: dict[str, str] | None = None,
    ):
        target = self_host.configure_aws.AwsTarget(
            account="123456789012",
            region="us-west-2",
            partition="aws",
            caller_arn=("arn:aws:sts::123456789012:assumed-role/Deploy/test-session"),
            profile="sandbox",
        )
        slack_payload = {
            "SLACK_CLIENT_ID": "123456789.987654321",
            "SLACK_CLIENT_SECRET": "a" * 32,
            "SLACK_SIGNING_SECRET": "b" * 32,
            "SLACK_APP_ID": "A123456789",
        }
        bridge_payload = {
            "BRIDGE_OAUTH_STATE_SECRET": "c" * 64,
            "BRIDGE_GATEWAY_JWT_PRIVATE_KEY_PEM": (
                "-----BEGIN PRIVATE KEY-----\nkey\n-----END PRIVATE KEY-----\n"
            ),
            "ADMIN_SECRET": "d" * 48,
        }
        existing_slack = self_host.SecretDocument("arn:slack", slack_payload, True)
        existing_bridge = self_host.SecretDocument("arn:bridge", bridge_payload, True)
        runner = mock.Mock()

        def run(command, **kwargs):
            if list(command) == ["bash", "scripts/resolve_agent_runtime.sh"]:
                return completed(
                    list(command),
                    "arn:aws:bedrock-agentcore:us-west-2:123456789012:"
                    "agent/00000000-0000-0000-0000-000000000001:1\n",
                )
            return completed(list(command))

        runner.run.side_effect = run
        arguments = [
            "--profile",
            "sandbox",
            "--region",
            "us-west-2",
            "--domain",
            "agent.example.com",
            "--certificate-arn",
            config().certificate_arn,
            "--skip-setup",
            "--yes",
        ]
        if skip_health_check:
            arguments.append("--skip-health-check")
        args = self_host.build_parser().parse_args(arguments)

        with (
            mock.patch.dict(
                os.environ, operator_environment or {}, clear=True
            ),
            mock.patch.object(self_host, "_require_tools"),
            mock.patch.object(
                self_host.configure_aws,
                "inspect_target",
                return_value=(target, "20"),
            ),
            mock.patch.object(
                self_host.configure_aws,
                "write_target",
                return_value=(Path("aws-targets.json"), "written"),
            ),
            mock.patch.object(
                self_host, "CommandRunner", return_value=runner
            ) as runner_factory,
            mock.patch.object(
                self_host,
                "write_generated_manifest",
                return_value=Path("manifest.json"),
            ),
            mock.patch.object(
                self_host,
                "read_secret",
                side_effect=[existing_slack, existing_bridge],
            ),
            mock.patch.object(
                self_host,
                "put_secret",
                side_effect=[existing_slack, existing_bridge],
            ),
            mock.patch.object(self_host, "_confirm"),
            mock.patch.object(
                self_host,
                "_pinned_agentcore_cli",
                return_value=nullcontext(Path("/tmp/pinned-agentcore")),
            ),
            mock.patch.object(
                self_host,
                "_existing_runtime_environment",
                return_value=existing_runtime_environment or {},
            ),
            mock.patch.object(
                self_host,
                "_existing_onboarding_environment",
                return_value=existing_onboarding_environment or {},
            ),
            mock.patch.object(self_host, "_cdk_deploy") as cdk_deploy,
            mock.patch.object(
                self_host,
                "_stack_outputs",
                return_value={"AlbDnsName": "alb.example.amazonaws.com"},
            ),
            mock.patch.object(
                self_host, "_write_state", return_value=Path("deployment.json")
            ),
            mock.patch.object(
                self_host, "_verify_health", return_value=health_result
            ) as verify_health,
        ):
            call_order = mock.Mock()
            call_order.attach_mock(cdk_deploy, "cdk_deploy")
            call_order.attach_mock(verify_health, "verify_health")
            call_order.attach_mock(runner.run, "run")
            yield {
                "args": args,
                "runner": runner,
                "runner_factory": runner_factory,
                "cdk_deploy": cdk_deploy,
                "verify_health": verify_health,
                "call_order": call_order,
            }

    def test_pinned_agentcore_cli_is_private_and_temporary(self) -> None:
        with self_host._pinned_agentcore_cli() as directory:
            executable = directory / "agentcore"
            self.assertTrue(executable.exists())
            self.assertEqual(stat.S_IMODE(executable.stat().st_mode), 0o700)
            self.assertIn("@aws/agentcore@0.24.1", executable.read_text())
        self.assertFalse(directory.exists())

    def test_guided_deploy_reuses_existing_interfaces_in_dependency_order(self) -> None:
        optional_environment = {
            "GITHUB_APP_ID": "3350219",
            "AGENTCORE_MEMORY_ID": "memory-existing",
            "AGENTCORE_SEMANTIC_STRATEGY_ID": "semantic-existing",
            "AGENTCORE_USER_PREF_STRATEGY_ID": "preferences-existing",
        }
        with self._guided_deploy(
            existing_runtime_environment=optional_environment,
            existing_onboarding_environment={"GITHUB_APP_SLUG": "slinck-ai"},
        ) as plan:
            result = self_host.deploy(plan["args"])

        self.assertEqual(result, 0)
        cdk_deploy = plan["cdk_deploy"]
        deployed_stacks = [call.args[1] for call in cdk_deploy.call_args_list]
        self.assertEqual(
            deployed_stacks,
            [
                (
                    "AgentCore-coreAgent-data-us-west-2",
                    "AgentCore-coreAgent-observability-us-west-2",
                ),
                ("AgentCore-coreAgent-services-us-west-2",),
                ("AgentCore-coreAgent-gateway-us-west-2",),
            ],
        )
        runner = plan["runner"]
        all_commands = [list(call.args[0]) for call in runner.run.call_args_list]
        self.assertIn(["bash", "scripts/deploy_agent.sh", "--yes"], all_commands)
        self.assertNotIn(["bash", "scripts/deploy_agent.sh"], all_commands)
        runner_factory = plan["runner_factory"]
        runtime_environment = runner_factory.call_args_list[1].args[0]
        self.assertEqual(runtime_environment["DOMAIN_NAME"], "agent.example.com")
        self.assertEqual(runtime_environment["DEPLOY_EXPERIMENTAL_SANDBOX"], "false")
        for name, value in optional_environment.items():
            self.assertEqual(runtime_environment[name], value)
        self.assertTrue(runtime_environment["PATH"].startswith("/tmp/pinned-agentcore"))
        services_call = cdk_deploy.call_args_list[1]
        self.assertIn(("githubAppId", "3350219"), services_call.args[3:])
        self.assertIn(("githubAppSlug", "slinck-ai"), services_call.args[3:])
        self.assertTrue(
            any("provision_gateway.py" in " ".join(command) for command in all_commands)
        )
        health_index = next(
            index
            for index, call in enumerate(plan["call_order"].mock_calls)
            if call[0] == "verify_health"
        )
        gateway_deploy_index = next(
            index
            for index, call in enumerate(plan["call_order"].mock_calls)
            if call[0] == "cdk_deploy"
            and call.args[1] == ("AgentCore-coreAgent-gateway-us-west-2",)
        )
        gateway_provision_index = next(
            index
            for index, call in enumerate(plan["call_order"].mock_calls)
            if call[0] == "run" and "provision_gateway.py" in " ".join(call.args[0])
        )
        self.assertLess(health_index, gateway_deploy_index)
        self.assertLess(gateway_deploy_index, gateway_provision_index)
        self.assertFalse(
            any("sandbox" in " ".join(command).lower() for command in all_commands)
        )

    def test_explicit_runtime_environment_overrides_preserved_value(self) -> None:
        with self._guided_deploy(
            skip_health_check=True,
            existing_runtime_environment={
                "GITHUB_APP_ID": "3350219",
                "AGENTCORE_MEMORY_ID": "memory-existing",
            },
            existing_onboarding_environment={"GITHUB_APP_SLUG": "slinck-ai"},
            operator_environment={
                "GITHUB_APP_ID": "",
                "AGENTCORE_MEMORY_ID": "memory-replacement",
                "GITHUB_APP_SLUG": "replacement-app",
            },
        ) as plan:
            self_host.deploy(plan["args"])

        runtime_environment = plan["runner_factory"].call_args_list[1].args[0]
        self.assertEqual(runtime_environment["GITHUB_APP_ID"], "")
        self.assertEqual(
            runtime_environment["AGENTCORE_MEMORY_ID"], "memory-replacement"
        )
        services_call = plan["cdk_deploy"].call_args_list[1]
        self.assertNotIn(("githubAppId", "3350219"), services_call.args[3:])
        self.assertNotIn(("githubAppSlug", "slinck-ai"), services_call.args[3:])
        self.assertIn(("githubAppSlug", "replacement-app"), services_call.args[3:])

    def test_health_failure_does_not_deploy_or_provision_gateway(self) -> None:
        with self._guided_deploy(health_result=False) as plan:
            with self.assertRaises(self_host.SelfHostError):
                self_host.deploy(plan["args"])

        deployed_stacks = [call.args[1] for call in plan["cdk_deploy"].call_args_list]
        self.assertNotIn(("AgentCore-coreAgent-gateway-us-west-2",), deployed_stacks)
        all_commands = [
            list(call.args[0]) for call in plan["runner"].run.call_args_list
        ]
        self.assertFalse(
            any("provision_gateway.py" in " ".join(command) for command in all_commands)
        )

    def test_skip_health_check_defers_gateway_until_rerun(self) -> None:
        output = io.StringIO()
        with self._guided_deploy(skip_health_check=True) as plan:
            with redirect_stdout(output):
                result = self_host.deploy(plan["args"])

        self.assertEqual(result, 0)
        plan["verify_health"].assert_not_called()
        deployed_stacks = [call.args[1] for call in plan["cdk_deploy"].call_args_list]
        self.assertNotIn(("AgentCore-coreAgent-gateway-us-west-2",), deployed_stacks)
        all_commands = [
            list(call.args[0]) for call in plan["runner"].run.call_args_list
        ]
        self.assertFalse(
            any("provision_gateway.py" in " ".join(command) for command in all_commands)
        )
        self.assertIn("Gateway deployment deferred", output.getvalue())
        self.assertIn("rerun without --skip-health-check", output.getvalue())

    def test_cdk_deploy_is_explicit_and_never_includes_sandbox(self) -> None:
        runner = mock.Mock()
        self_host._cdk_deploy(
            runner,
            ("AgentCore-coreAgent-services-us-west-2",),
            config(),
            ("agentRuntimeArn", "arn:runtime"),
            ("domainName", "agent.example.com"),
        )
        command = runner.run.call_args.args[0]
        self.assertEqual(command[:3], ["npx", "cdk", "deploy"])
        self.assertIn("--require-approval", command)
        self.assertIn("never", command)
        self.assertIn("agentRuntimeArn=arn:runtime", command)
        self.assertNotIn("sandbox", " ".join(command).lower())
        self.assertEqual(
            runner.run.call_args.kwargs["cwd"], self_host.ROOT / "infra/data"
        )

    def test_typed_confirmation_must_match_account_and_region(self) -> None:
        with (
            mock.patch("builtins.input", return_value="yes"),
            self.assertRaisesRegex(self_host.SelfHostError, "cancelled"),
        ):
            self_host._confirm(config(), assume_yes=False)
        with mock.patch("builtins.input", return_value="deploy 123456789012/us-west-2"):
            self_host._confirm(config(), assume_yes=False)

    def test_dry_run_is_non_mutating_and_names_every_phase(self) -> None:
        output = io.StringIO()
        args = self_host.build_parser().parse_args(
            ["--dry-run", "--domain", "agent.example.com", "--region", "us-west-2"]
        )
        with redirect_stdout(output):
            result = self_host._dry_run(args)
        self.assertEqual(result, 0)
        text = output.getvalue()
        for phrase in (
            "Slack manifest",
            "Secrets Manager",
            "AgentCore runtime",
            "bridge/onboarding",
            "tenant-scoped AgentCore Gateway",
            "Experimental PR sandbox: disabled",
        ):
            self.assertIn(phrase, text)

    def test_state_file_contains_no_secret_material(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(self_host, "STATE_DIR", Path(temporary)):
                destination = self_host._write_state(
                    config(), {"AlbDnsName": "alb.example.amazonaws.com"}
                )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(payload["alb_dns_name"], "alb.example.amazonaws.com")
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
            serialized = json.dumps(payload).lower()
            self.assertNotIn("secret", serialized)
            self.assertNotIn("token", serialized)


if __name__ == "__main__":
    unittest.main()
