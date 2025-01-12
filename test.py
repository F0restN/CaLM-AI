import site
import os
import sys

# print(site.getsitepackages())

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
# print("project_path: ", project_path)

site.addsitedir(project_path)

# Verify the path is added

# for path in sys.path:
#     print("Path: ", path)