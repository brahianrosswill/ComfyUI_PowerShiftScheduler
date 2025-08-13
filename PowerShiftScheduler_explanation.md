# PowerShiftScheduler Node Explained

This document provides a detailed explanation of the PowerShiftScheduler node and its parameters, based on the code in `__init__.py`.

## PowerShiftSchedulerNode

The `PowerShiftSchedulerNode` is a custom scheduling node for ComfyUI that allows for more control over the distribution of sampling steps. It uses a power function to distribute the steps, which can be further adjusted with a midpoint shift.

### `INPUT_TYPES`

The `INPUT_TYPES` class method defines the parameters that the user can set in the ComfyUI interface.

```python
@classmethod
def INPUT_TYPES(s):
    return {"required":
                {"model": ("MODEL",),
                 "steps": ("INT", {"default": 20, "min": 3, "max": 1000}),
                 "power": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 5.0, "step": 0.001}),
                 "midpoint_shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.001}),
                 "discard_penultimate": ("BOOLEAN", {"default": False}),
                 "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                  }
           }
```

Here is a detailed breakdown of each parameter:

*   **`model`**:
    *   **Type**: `MODEL`
    *   **Description**: This is the input model that will be used for the sampling process. It's a standard input for most sampling-related nodes in ComfyUI.

*   **`steps`**:
    *   **Type**: `INT`
    *   **Default**: `20`
    *   **Min/Max**: `3` / `1000`
    *   **Description**: This parameter determines the number of sampling steps to be performed. A higher number of steps will generally result in a more detailed and higher-quality image, but it will also take longer to generate.

*   **`power`**:
    *   **Type**: `FLOAT`
    *   **Default**: `2.0`
    *   **Min/Max**: `0.0` / `5.0`
    *   **Step**: `0.001`
    *   **Description**: This parameter controls the curvature of the step distribution. It applies a power function to the normalized timestep distribution.
        *   A `power` value of `1.0` results in a linear distribution of steps.
        *   A `power` value greater than `1.0` (e.g., the default of `2.0`) concentrates the steps more towards the beginning and end of the sampling process, with fewer steps in the middle. This can be useful for quickly establishing the main composition and then refining the details.
        *   A `power` value less than `1.0` will have the opposite effect, concentrating the steps in the middle of the sampling process.

*   **`midpoint_shift`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min/Max**: `0.0` / `5.0`
    *   **Step**: `0.001`
    *   **Description**: This parameter adjusts the midpoint of the step distribution.
        *   A `midpoint_shift` of `1.0` has no effect on the distribution.
        *   A value greater than `1.0` shifts the concentration of steps towards the end of the sampling process. This can be useful for spending more time on refining the details of the image.
        *   A value less than `1.0` shifts the concentration of steps towards the beginning of the sampling process. This can be useful for quickly establishing the overall composition and structure of the image.

*   **`discard_penultimate`**:
    *   **Type**: `BOOLEAN`
    *   **Default**: `False`
    *   **Description**: When set to `True`, this parameter discards the second-to-last sigma value from the scheduler's output. In some cases, the penultimate step can introduce a small amount of noise or undesirable artifacts into the final image. Enabling this option can help to mitigate this issue and produce a cleaner result.

*   **`denoise`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min/Max**: `0.0` / `1.0`
    *   **Step**: `0.01`
    *   **Description**: This parameter controls the amount of denoising to be applied. It is a way to control the denoising process. If `denoise` is less than `1.0`, the effective number of steps is increased (`total_steps = int(steps/denoise)`), but only the last `steps` number of sigmas are returned. This allows for a more refined denoising process over a larger number of steps, while still outputting the desired number of sigmas.

## `power_shift_scheduler` function

This is the core function that generates the sigma values based on the input parameters.

```python
def power_shift_scheduler(model_sampling, steps, power=2.0, midpoint_shift=1.0, discard_penultimate=False):
    total_timesteps = (len(model_sampling.sigmas) - 1)
    x = numpy.linspace(0, 1, steps, endpoint=False)
    x = x**midpoint_shift

    ts_normalized = (1 - x**power)**power
    ts = numpy.rint(ts_normalized * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        t_int = min(int(t), total_timesteps)
        if t_int != last_t:
            sigs.append(float(model_sampling.sigmas[t_int]))
        last_t = t_int

    sigs.append(0.0)
    if discard_penultimate is True:
        sigmas = torch.FloatTensor(sigs)
        return torch.cat((sigmas[:-2], sigmas[-1:]))
    else:
        return torch.FloatTensor(sigs)
```

### How it works:

1.  **`total_timesteps`**: It gets the total number of timesteps from the model's sampling sigmas.
2.  **`numpy.linspace`**: It creates a linearly spaced array of `steps` numbers from 0 to 1.
3.  **`midpoint_shift`**: The `midpoint_shift` is applied to this array, which shifts the distribution of the steps.
4.  **`power`**: The `power` parameter is then applied twice, once to the inverted array and then to the result. This creates the characteristic power-curved distribution.
5.  **`ts`**: The normalized timesteps are then scaled to the total number of timesteps of the model.
6.  **Sigma Selection**: The function then iterates through the calculated timesteps (`ts`) and selects the corresponding sigma values from the model's `sigmas`. It also ensures that no duplicate sigmas are added.
7.  **`discard_penultimate`**: If `discard_penultimate` is `True`, the second-to-last sigma is removed.
8.  **Return Value**: The function returns a `torch.FloatTensor` of the selected sigma values.
