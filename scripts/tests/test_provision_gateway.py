from __future__ import annotations

import copy
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
AWS_SCRIPTS = ROOT / "infra/data/scripts"
sys.path.insert(0, str(AWS_SCRIPTS))

# The local-tooling suite intentionally has no AWS SDK dependency. Stub only
# the imports needed to load the script; every SDK interaction is mocked below.
try:
    import boto3  # noqa: F401
except ModuleNotFoundError:
    boto3_module = types.ModuleType("boto3")
    boto3_module.client = mock.Mock()  # type: ignore[attr-defined]
    sys.modules["boto3"] = boto3_module

try:
    from botocore.exceptions import ClientError  # noqa: F401
except ModuleNotFoundError:
    botocore_module = types.ModuleType("botocore")
    exceptions_module = types.ModuleType("botocore.exceptions")

    class ClientError(Exception):
        pass

    exceptions_module.ClientError = ClientError  # type: ignore[attr-defined]
    sys.modules["botocore"] = botocore_module
    sys.modules["botocore.exceptions"] = exceptions_module

_previous_region = os.environ.get("AWS_REGION")
os.environ["AWS_REGION"] = "us-west-2"
try:
    import provision_gateway
finally:
    if _previous_region is None:
        os.environ.pop("AWS_REGION", None)
    else:
        os.environ["AWS_REGION"] = _previous_region


def stack_outputs() -> dict[str, str]:
    return {
        "InterceptorLambdaArn": "arn:aws:lambda:us-west-2:123:function:interceptor",
        "GatewayRoleArn": "arn:aws:iam::123:role/gateway",
        "BridgeJwksUrl": "https://agent.example.com/jwks.json",
        "BridgeOidcDiscoveryUrl": (
            "https://agent.example.com/.well-known/openid-configuration"
        ),
    }


def ready_gateway() -> dict[str, object]:
    outputs = stack_outputs()
    return {
        "gatewayId": "gateway-1",
        "gatewayArn": "arn:aws:bedrock-agentcore:us-west-2:123:gateway/gateway-1",
        "gatewayUrl": "https://gateway.example.com/mcp",
        "status": "READY",
        "roleArn": outputs["GatewayRoleArn"],
        "protocolType": "MCP",
        "authorizerType": "CUSTOM_JWT",
        "authorizerConfiguration": {
            "customJWTAuthorizer": {
                "discoveryUrl": outputs["BridgeOidcDiscoveryUrl"],
                "allowedAudience": ["agentcore-gateway"],
            }
        },
        "interceptorConfigurations": [
            {
                "interceptor": {
                    "lambda": {"arn": outputs["InterceptorLambdaArn"]}
                },
                "interceptionPoints": ["REQUEST"],
                "inputConfiguration": {"passRequestHeaders": True},
            }
        ],
    }


class GatewayConfigurationTests(unittest.TestCase):
    def test_supported_existing_gateway_has_no_differences(self) -> None:
        self.assertEqual(
            provision_gateway.gateway_configuration_differences(
                ready_gateway(), stack_outputs()
            ),
            [],
        )

    def test_security_sensitive_drift_is_reported(self) -> None:
        mutations = {
            "role": lambda gateway: gateway.update({"roleArn": "arn:wrong"}),
            "issuer": lambda gateway: gateway["authorizerConfiguration"][
                "customJWTAuthorizer"
            ].update({"discoveryUrl": "https://wrong.example.com"}),
            "audience": lambda gateway: gateway["authorizerConfiguration"][
                "customJWTAuthorizer"
            ].update({"allowedAudience": ["other"]}),
            "interceptor": lambda gateway: gateway["interceptorConfigurations"][
                0
            ]["interceptor"]["lambda"].update({"arn": "arn:wrong"}),
            "headers": lambda gateway: gateway["interceptorConfigurations"][0][
                "inputConfiguration"
            ].update({"passRequestHeaders": False}),
            "url": lambda gateway: gateway.update({"gatewayUrl": ""}),
        }
        for name, mutate in mutations.items():
            gateway = copy.deepcopy(ready_gateway())
            mutate(gateway)
            with self.subTest(name=name):
                self.assertTrue(
                    provision_gateway.gateway_configuration_differences(
                        gateway, stack_outputs()
                    )
                )

    def test_existing_gateway_drift_fails_without_publishing_ssm(self) -> None:
        gateway = ready_gateway()
        gateway["roleArn"] = "arn:wrong"
        client = mock.Mock()
        client.get_gateway.return_value = gateway
        with (
            mock.patch.object(
                provision_gateway, "fetch_stack_outputs", return_value=stack_outputs()
            ),
            mock.patch.object(provision_gateway.boto3, "client", return_value=client),
            mock.patch.object(
                provision_gateway,
                "find_existing_gateway",
                return_value={"gatewayId": "gateway-1", "status": "READY"},
            ),
            mock.patch.object(provision_gateway, "publish_to_ssm") as publish,
        ):
            result = provision_gateway.main(["--region", "us-west-2"])

        self.assertEqual(result, 1)
        publish.assert_not_called()

    def test_existing_gateway_rerun_waits_then_republishes(self) -> None:
        client = mock.Mock()
        client.get_gateway.return_value = {"gatewayId": "gateway-1", "status": "UPDATING"}
        with (
            mock.patch.object(
                provision_gateway, "fetch_stack_outputs", return_value=stack_outputs()
            ),
            mock.patch.object(provision_gateway.boto3, "client", return_value=client),
            mock.patch.object(
                provision_gateway,
                "find_existing_gateway",
                return_value={"gatewayId": "gateway-1", "status": "UPDATING"},
            ),
            mock.patch.object(
                provision_gateway, "wait_for_gateway_ready", return_value=ready_gateway()
            ) as wait,
            mock.patch.object(provision_gateway, "publish_to_ssm") as publish,
        ):
            result = provision_gateway.main(["--region", "us-west-2"])

        self.assertEqual(result, 0)
        wait.assert_called_once_with(client, "gateway-1")
        publish.assert_called_once_with(
            "us-west-2",
            gateway_id="gateway-1",
            gateway_url="https://gateway.example.com/mcp",
            gateway_arn="arn:aws:bedrock-agentcore:us-west-2:123:gateway/gateway-1",
        )

    def test_existing_gateway_dry_run_does_not_publish_ssm(self) -> None:
        client = mock.Mock()
        client.get_gateway.return_value = ready_gateway()
        with (
            mock.patch.object(
                provision_gateway, "fetch_stack_outputs", return_value=stack_outputs()
            ),
            mock.patch.object(provision_gateway.boto3, "client", return_value=client),
            mock.patch.object(
                provision_gateway,
                "find_existing_gateway",
                return_value={"gatewayId": "gateway-1", "status": "READY"},
            ),
            mock.patch.object(provision_gateway, "publish_to_ssm") as publish,
        ):
            result = provision_gateway.main(
                ["--region", "us-west-2", "--dry-run"]
            )

        self.assertEqual(result, 0)
        publish.assert_not_called()

    def test_update_unsuccessful_is_terminal_without_polling_timeout(self) -> None:
        client = mock.Mock()
        client.get_gateway.return_value = {
            "gatewayId": "gateway-1",
            "status": "UPDATE_UNSUCCESSFUL",
            "statusReasons": ["invalid update"],
        }

        with self.assertRaises(SystemExit):
            provision_gateway.wait_for_gateway_ready(
                client, "gateway-1", timeout=120
            )

        client.get_gateway.assert_called_once_with(gatewayIdentifier="gateway-1")


if __name__ == "__main__":
    unittest.main()
