from pathlib import Path

# the path to this file
path = Path(__file__)

# full path, resolve call handles things like ../
print(path.resolve().absolute())

# just this file
print(path.name)

# file extension
print(path.suffix)

# each part of the path split up as an array
print(path.parts)

# Current working directory
print(path.cwd())

# checks if the path exists
print(path.exists())

# fill in more as needed...

# unlink deletes a file, rmdir removes a directory
# print(path.unlink())
