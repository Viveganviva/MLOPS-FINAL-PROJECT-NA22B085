Use only my existing conda environment named ai_env at:
C:\Users\vvads\.conda\envs\ai_env

Do not create .venv or any new environment.
Do not edit .vscode/settings.json unless I ask.
Assume the notebook/kernel and terminal must both use ai_env.

Before any code change or install:
1) confirm the selected interpreter/kernel is ai_env
2) use the active ai_env terminal for shell commands
3) install missing packages only into ai_env
4) if something is missing, tell me the exact command first and wait

For notebooks:
- use the ai_env kernel only

For debugging:
- run and debug with the ai_env interpreter only

If you need to inspect packages, use:
python -m pip list
conda list

If you need to install packages, use:
conda install ... or python -m pip install ...
inside the active ai_env terminal only.

Working of code is the priority.