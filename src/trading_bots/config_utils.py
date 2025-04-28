import os
import json
import logging
from typing import Optional, Dict, cast

logger = logging.getLogger(__name__)


def load_secrets_from_aws(
    secret_name: str, region_name: str
) -> Optional[Dict[str, str]]:
    """Loads secrets from AWS Secrets Manager."""
    try:
        # Lazy import boto3 only if needed
        import boto3
        from botocore.exceptions import (
            ClientError,
            NoCredentialsError,
            PartialCredentialsError,
        )

        logger.info(
            f"Attempting to load secrets from AWS Secrets Manager: {secret_name} in {region_name}"
        )
        session = boto3.session.Session()
        client = session.client(service_name="secretsmanager", region_name=region_name)

        get_secret_value_response = client.get_secret_value(SecretId=secret_name)

        if "SecretString" in get_secret_value_response:
            secret_string = get_secret_value_response["SecretString"]
            secrets = cast(Dict[str, str], json.loads(secret_string))
            logger.info(f"Successfully loaded secrets from AWS: {secret_name}")
            # Expecting keys like 'api_key' and 'secret_key' within the secret JSON
            return secrets
        else:
            # Handle binary secrets if necessary (decode from base64)
            logger.warning(f"Secret '{secret_name}' does not contain a SecretString.")
            return None

    except (NoCredentialsError, PartialCredentialsError):
        logger.error(
            "AWS credentials not found. Ensure credentials are configured (e.g., environment variables, instance profile).",
            exc_info=True,
        )
        return None
    except ClientError as e:
        logger.error(
            f"AWS ClientError loading secret '{secret_name}': {e}", exc_info=True
        )
        # Handle specific errors like ResourceNotFoundException if needed
        return None
    except ImportError:
        logger.error("boto3 library is not installed. Cannot use AWS Secrets Manager.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON secret string from '{secret_name}': {e}")
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred loading secrets from AWS: {e}", exc_info=True
        )
        return None


# Example of how to structure the secret in AWS Secrets Manager (as JSON):
# {
#   "BINANCE_API_KEY": "your_actual_live_key",
#   "BINANCE_SECRET_KEY": "your_actual_live_secret",
#   "BINANCE_TESTNET_API_KEY": "your_actual_testnet_key",
#   "BINANCE_TESTNET_SECRET_KEY": "your_actual_testnet_secret"
# }
