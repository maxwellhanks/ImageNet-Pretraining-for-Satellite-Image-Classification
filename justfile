set windows-shell := ["powershell", "-NoProfile", "-Command"]

RM_R := if os_family() == "windows" { "Remove-Item -Recurse" } else { "rm -r" }
ON_ERROR_CONTINUE := if os_family() == "windows" { '; $ErrorActionPreference = "Continue"' } else { "|| true" }

DEFAULT_VENV_NAME := ".venv"

# interactive menu to select a command
_default:
	@ just --choose || :

alias default := _default

# install all dependencies and sync the virtual environment
install:
	uv sync --all-groups

# list registered models
list-models:
	uv run satbench list-models

# list local checkpoints (joined with results CSV metadata when available)
list-checkpoints:
	uv run satbench list-checkpoints

# download + cache a dataset (eurosat_rgb | uc_merced_rgb | fake)
[arg("dataset", pattern="eurosat_rgb|uc_merced_rgb|fake")]
prepare dataset="eurosat_rgb":
	uv run satbench prepare-data --dataset {{dataset}}

# download + cache all datasets (eurosat_rgb, uc_merced_rgb, fake)
prepare-all:
	uv run satbench prepare-data --dataset "eurosat_rgb"
	uv run satbench prepare-data --dataset "uc_merced_rgb"
	uv run satbench prepare-data --dataset "fake"

# quick smoke experiment
[arg("user", pattern=".+")]
smoke user="":
	uv run satbench run --config configs/smoke_test.yaml --user {{user}}

# transfer-eval every local checkpoint onto another dataset
#   just transfer-all manuel
#   just transfer-all manuel uc_merced_rgb
[arg("user", pattern=".+")]
[arg("target", pattern="uc_merced_rgb")]
transfer-all user="" target="uc_merced_rgb":
	uv run satbench transfer-eval --user {{user}} --target {{target}} --all

# transfer-eval by training filters (no need to copy a config hash)
#   just transfer manuel resnet50_imagenet 1.0 0
#   just transfer manuel resnet50_imagenet 0.1 0 uc_merced_rgb
[arg("user", pattern=".+")]
[arg("model", pattern=".+")]
[arg("fraction", pattern=".+")]
[arg("seed", pattern=".+")]
[arg("target", pattern="uc_merced_rgb")]
transfer user="" model="" fraction="" seed="" target="uc_merced_rgb":
	uv run satbench transfer-eval --user {{user}} --target {{target}} --model {{model}} --fraction {{fraction}} --seed {{seed}}

# offline unit tests (synthetic fake dataset)
test:
	uv run pytest

# convert notebook to pdf
[arg("input", pattern=".+")]
topdf input="" output=(file_stem(input)):
	uv run --with "nbconvert[webpdf]" jupyter nbconvert --to webpdf --allow-chromium-download --no-prompt {{input}} --output {{output}}

# package tracked repository files into a zip file
zip output="../stat3007-7007_g3_code.zip":
	git archive --format=zip --prefix=stat3007-7007_g3_project/ -o {{output}} HEAD


# create a fresh virtual environment (defaults to .venv)
venv name=DEFAULT_VENV_NAME:
	uv venv {{ name }}

# wipe downloaded data + uint8 caches
clean-data:
	{{ RM_R }} data 			{{ ON_ERROR_CONTINUE }}

# remove the .venv and .pytest_cache folder and clear uv's internal cache
clean:
	uv clean
	{{ RM_R }} .venv 			{{ ON_ERROR_CONTINUE }}
	{{ RM_R }} .pytest_cache	{{ ON_ERROR_CONTINUE }}