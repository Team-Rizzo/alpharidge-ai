import os

# The suite is hermetic: no checkout env file (.vali_env, .miner_env, which alpharidge_ai.config
# loads over the environment) and no EL_ setting from the caller's shell reaches it, so tests
# price at today's pay, the default. Tests of env files and pay settings set their own.
os.environ["ALPHARIDGE_SKIP_ENV_FILES"] = "1"
for _k in [k for k in os.environ if k.startswith("EL_")]:
    del os.environ[_k]
