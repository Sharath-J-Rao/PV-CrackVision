import unittest
from unittest import mock

try:
    from training.install_package_dependencies import collect_requirements, installed_constraints, isolated_pip_environment
except ModuleNotFoundError:
    from install_package_dependencies import collect_requirements, installed_constraints, isolated_pip_environment


class InstallPackageDependenciesTests(unittest.TestCase):
    @mock.patch('training.install_package_dependencies.metadata.requires')
    def test_collects_required_dependencies_and_excludes_protected_runtime(self, requires):
        requires.side_effect = [
            [
                'filelock',
                'typing-extensions>=4.10',
                'torch>=2.0',
                'torchvision>=0.15',
                'platformdirs; sys_platform == "win32"',
                'not-on-this-platform; sys_platform == "made-up"',
                'optional-gpu-package; extra == "gpu"',
                'base-extra-package; extra == ""',
            ],
            ['numpy>=1.23', 'Pillow>=9'],
        ]
        result = collect_requirements(
            ['torch', 'torchvision'],
            {'torch', 'torchvision', 'torchaudio'},
        )
        self.assertIn('filelock', result)
        self.assertIn('typing-extensions>=4.10', result)
        self.assertIn('numpy>=1.23', result)
        self.assertIn('Pillow>=9', result)
        self.assertNotIn('torch>=2.0', result)
        self.assertNotIn('torchvision>=0.15', result)
        self.assertNotIn('not-on-this-platform; sys_platform == "made-up"', result)
        self.assertNotIn('optional-gpu-package; extra == "gpu"', result)
        self.assertIn('base-extra-package; extra == ""', result)

    @mock.patch('training.install_package_dependencies.metadata.version')
    def test_constraints_pin_exact_installed_versions(self, version):
        version.side_effect = ['2.9.0+cpu', '0.24.0+cpu']
        self.assertEqual(
            installed_constraints(['torch', 'torchvision']),
            ['torch==2.9.0+cpu', 'torchvision==0.24.0+cpu'],
        )

    def test_pip_environment_ignores_user_indexes_and_configuration_files(self):
        environment = isolated_pip_environment({
            'PIP_INDEX_URL': 'https://invalid.example/simple',
            'PIP_EXTRA_INDEX_URL': 'https://also-invalid.example/simple',
            'PIP_FIND_LINKS': 'C:/wheels',
            'PIP_CONFIG_FILE': 'C:/Users/example/pip.ini',
            'PATH': 'C:/Windows',
        })
        self.assertNotIn('PIP_INDEX_URL', environment)
        self.assertNotIn('PIP_EXTRA_INDEX_URL', environment)
        self.assertNotIn('PIP_FIND_LINKS', environment)
        self.assertEqual(environment['PIP_CONFIG_FILE'], __import__('os').devnull)
        self.assertEqual(environment['PATH'], 'C:/Windows')


if __name__ == '__main__':
    unittest.main()
