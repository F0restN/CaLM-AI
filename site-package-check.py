import site
import os
import sys

site_packages = site.getsitepackages()

custom_path = os.path.abspath(os.path.join(os.getcwd(), 'src'))

if custom_path in sys.path:
    print("Environment set up !")
else:
    sitepackage_file_path = os.path.join(site_packages[0], 'calm_adrd_src_paths.pth')
    with open(sitepackage_file_path, 'w') as f:
        f.write(custom_path)
    print("Environment set up !")

for path in sys.path:
    print(path)




