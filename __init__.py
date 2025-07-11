import torch
import numpy
from comfy.samplers import SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES

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

class PowerShiftSchedulerNode:
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
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, power, midpoint_shift, discard_penultimate, denoise):
        total_steps = steps
        if denoise < 1.0:
            total_steps = int(steps/denoise)

        sigmas = power_shift_scheduler(model.get_model_object("model_sampling"), total_steps, power, midpoint_shift, discard_penultimate=discard_penultimate).cpu()
        sigmas = sigmas[-(steps + 1):]

        return (sigmas, )

scheduler_name = "power_shift"
if scheduler_name not in SCHEDULER_HANDLERS:
    scheduler_handler = SchedulerHandler(handler=power_shift_scheduler, use_ms=True)
    SCHEDULER_HANDLERS[scheduler_name] = scheduler_handler
    if scheduler_name not in SCHEDULER_NAMES:
        SCHEDULER_NAMES.append(scheduler_name)

NODE_CLASS_MAPPINGS = {
    "PowerShiftScheduler": PowerShiftSchedulerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PowerShiftScheduler": "Power Shift Scheduler",
}
