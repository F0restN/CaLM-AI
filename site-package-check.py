import site
import os
import sys

site_packages = site.getsitepackages()

# print(site_packages)

# custom_path = os.path.join(site_packages, 'drake_custom_paths.pth')

# with open(custom_path, 'w') as f:
#     f.write('/Users/drakezhou/Development-Projects/CaLM-ADRD/calm-adrd-rag/')

for path in sys.path:
    print(path)