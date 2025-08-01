from pathlib import Path

# the path to this file
path = Path(__file__)

# full path, resolve call handles things like ../
print(path.resolve().absolute())

# just this file/directory
print(path.name)

# file extension
print(path.suffix)

# each part of the path split up as an array
print(path.parts)

# Current working directory
print(path.cwd())

# checks if the path exists
print(path.exists())

# checks if the path is file
print(path.is_file())

# checks if the path is a directory
print(path.is_dir())

# creates the current directory
path.mkdir(parents=True, exist_ok=True)

# fill in more as needed...

# unlink deletes a file, rmdir removes a directory
# print(path.unlink())
