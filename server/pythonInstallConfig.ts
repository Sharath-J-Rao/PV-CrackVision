/** Reproducible package-index configuration for the managed Python runtime. */
export const PYTORCH_CPU_INDEX_URL = 'https://download.pytorch.org/whl/cpu';
export const PYPI_INDEX_URL = 'https://pypi.org/simple';

/**
 * Install only official binary CPU wheels for torch/torchvision.
 * --no-deps keeps their transitive dependencies out of the PyTorch-only index;
 * --only-binary prevents an accidental source build from that index.
 */
export function buildTorchCpuWheelInstallArgs(): string[] {
  return buildTorchWheelInstallArgs('cpu');
}

/** PyTorch wheel flavours: CPU, or an official CUDA build (GPU training). */
export type TorchVariant = 'cpu' | 'cu118' | 'cu124' | 'cu126' | 'cu128';
const TORCH_VARIANTS: readonly TorchVariant[] = ['cpu', 'cu118', 'cu124', 'cu126', 'cu128'];
export const torchIndexUrl = (variant: TorchVariant): string => `https://download.pytorch.org/whl/${variant}`;

/**
 * Pick the PyTorch wheel for this machine. Why: the app previously forced CPU-only
 * wheels, so a GPU laptop could never train on its GPU.
 * - PV_TORCH_VARIANT (cpu|cu118|cu124|cu126|cu128) overrides detection.
 * - Otherwise `nvidia-smi` output is parsed for the driver's max CUDA version and the newest
 *   wheel the driver supports is chosen (cu128 is required for RTX 50-series).
 * - No NVIDIA driver (or unparsable output) -> cpu, so CPU-only laptops keep working.
 */
export function chooseTorchVariant(override: string | undefined, nvidiaSmiOutput: string | undefined): TorchVariant {
  const forced = override?.trim().toLowerCase();
  if (forced && forced !== 'auto') {
    if ((TORCH_VARIANTS as readonly string[]).includes(forced)) return forced as TorchVariant;
    throw new Error(`PV_TORCH_VARIANT must be auto or one of ${TORCH_VARIANTS.join(', ')}; got "${override}".`);
  }
  const match = nvidiaSmiOutput?.match(/CUDA Version:\s*(\d+)\.(\d+)/i);
  if (!match) return 'cpu';
  const version = Number(match[1]) + Number(match[2]) / 100;
  if (version >= 12.08) return 'cu128';
  if (version >= 12.06) return 'cu126';
  if (version >= 12.04) return 'cu124';
  if (version >= 11.08) return 'cu118';
  return 'cpu';
}

export function buildTorchWheelInstallArgs(variant: TorchVariant, forceReinstall = false): string[] {
  return [
    '-m', 'pip', 'install', '--disable-pip-version-check',
    '--no-deps', '--only-binary=:all:',
    ...(forceReinstall ? ['--force-reinstall'] : []),
    'torch', 'torchvision',
    '--index-url', variant === 'cpu' ? PYTORCH_CPU_INDEX_URL : torchIndexUrl(variant),
  ];
}

/** Install a package wheel without allowing pip to replace protected dependencies. */
export function buildPyPiWheelInstallArgs(...packages: string[]): string[] {
  if (!packages.length || packages.some((name) => !name.trim())) {
    throw new Error('At least one non-empty Python package name is required.');
  }
  return [
    '-m', 'pip', 'install', '--disable-pip-version-check',
    '--no-deps', '--only-binary=:all:',
    ...packages,
    '--index-url', PYPI_INDEX_URL,
  ];
}

/**
 * Run the metadata-driven dependency installer for already-installed wheels.
 * The helper reads Requires-Dist from each wheel and pins protected package
 * versions so pip cannot silently swap the CPU PyTorch build.
 */
export function buildPackageDependencyInstallArgs(
  helperPath: string,
  packages: string[],
  protectedPackages: string[] = ['torch', 'torchvision'],
): string[] {
  if (!helperPath.trim() || !packages.length || packages.some((name) => !name.trim())) {
    throw new Error('A helper path and at least one installed package are required.');
  }
  return [helperPath, ...packages, '--preserve', ...protectedPackages];
}
