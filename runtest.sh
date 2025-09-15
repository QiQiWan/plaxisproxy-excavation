# MACOS
PYTHONPATH="$(pwd)/src" python -m unittest discover -s tests/structures -p "test_*.py" -v

# windows
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"

python -m unittest discover -s tests -p "test_*.py" -v
# components
python -m unittest discover -s tests/components -p "test_*.py" -v
# core
python -m unittest discover -s tests/core -p "test_*.py" -v
# materials
python -m unittest discover -s tests/materials -p "test_*.py" -v
# Plaxishelper
python -m unittest discover -s tests/Plaxishelper -p "test_*.py" -v
# structures
python -m unittest discover -s tests/structures -p "test_*.py" -v


# Calcuate the number of code lines
git ls-files | xargs cat | wc -l

git ls-files src/ | xargs cat | wc -l