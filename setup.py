from distutils.core import setup
setup(
  name = 'HYBparsimony',         # How you named your package folder (MyLib)
  packages = ['HYBparsimony'],   # Chose the same as "name"
  version = '0.0.1',      # Start with a small number and increase it with every change you make
  python_requires='>=3.9',
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'TYPE YOUR DESCRIPTION HERE',   # Give a short description about your library
  author = 'Jose Divasón',                   # Type in your name
  author_email = 'jose.divason@unirioja.es',      # Type in your E-Mail
  url = 'https://github.com/jodivaso/hyb-parsimony',   # Provide either the link to your github or to your website
  #download_url = 'https://github.com/jodivaso/hyb-parsimony/archive/refs/tags/0.0.2.tar.gz',
  keywords = ['SOME', 'MEANINGFULL', 'KEYWORDS'],   # Keywords that define your package best
  install_requires=[
      'numpy',
      'pandas>=2.0',
      'scikit-learn',
      'seaborn',
      'matplotlib'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.9',
  ],
)