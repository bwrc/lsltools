from setuptools import setup

setup(name='lsltools',
      version='0.1',
      description='LSLtools is a collection of utlities for working with the lab streaming layer.',
      author='Jari Torniainen, Andreas Henelius, Brain Work Research Center at the Finnish Institute of Occupational Health',
      author_email='jari.torniainen@ttl.fi, andreas.henelius@ttl.fi',
      url='http://github.io/FIOH-BWRC/lsltools',
      license='MIT',
      packages=['lsltools'],
      package_dir = {'lsltools': 'lsltools'},
      package_data={'lsltools' : ['lsltools/liblsl32.dll','lsltools/liblsl32.dylib','lsltools/liblsl64.dll', 'lsltools/liblsl64.dylib', 'lsltools/liblsl64.so']},
      include_package_data=True,


)



