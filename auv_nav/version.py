"""auv_nav package version"""
import os.path

version = ""

# Append annotation to version string to indicate development versions.
#
# An empty (modulo comments and blank lines) commit_hash.txt is used
# to indicate a release, in which case nothing is appended to version
# string as defined above.
path_to_hashfile = os.path.join(os.path.dirname(__file__), "commit_hash.txt")
if os.path.exists(path_to_hashfile):
    commit_version = ""
    with open(path_to_hashfile, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                # Ignore blank lines and comments, the latter being
                # any line that begins with #.
                continue

            # First non-blank line is assumed to be the commit hash
            commit_version = line
            break

    if len(commit_version) > 0:
        version = commit_version
else:
    version += ".dev0+unknown.commit"