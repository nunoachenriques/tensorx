# Copyright 2016 Davide Nunes

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================================================================================================

from distutils.core import setup


_VERSION = '0.1.0'

setup(
    name='tensorx',
    version=_VERSION,
    packages=['tensorx', 'tensorx.test', 'tensorx.parts'],
    url='https://github.com/davidelnunes/tensorx',
    license='Apache 2.0',
    author='Davide Nunes',
    author_email='davidelnunes@gmail.com',
    description='Utility Library for TensorFlow',

    install_requires=[
        'networkx>=1.11',
        'numpy >= 1.8.2',
        'tensorflow >= 0.6.0'],

    dependency_links=['https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp34-none-linux_x86_64.whl']
)
