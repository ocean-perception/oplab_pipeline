"""auv_nav package version"""
import os.path

version_info = (0, 0, 2)

version = '.'.join([str(x) for x in version_info])


# Append annotation to version string to indicate development versions.
#
# An empty (modulo comments and blank lines) commit_hash.txt is used
# to indicate a release, in which case nothing is appended to version
# string as defined above.
path_to_hashfile = os.path.join(os.path.dirname(__file__), "commit_hash.txt")
if os.path.exists(path_to_hashfile):
    commit_hash = ""
    with open(path_to_hashfile, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                # Ignore blank lines and comments, the latter being
                # any line that begins with #.
                continue

            # First non-blank line is assumed to be the commit hash
            commit_hash = line
            break

    if len(commit_hash) > 0:
        version += ".dev0+" + commit_hash
else:
    version += ".dev0+unknown.commit"