# Saving the solution during the algorithm

At times, it can be useful to save the current iterate in the algorithm every now and then. For this, the `SaveSettings` struct can be used. It can be provided to `solvesdp` using the `save_settings` keyword. This allows to save the solution of the algorithm
  - every k'th iteration, when `iter_interval = k`,
  - every s seconds, when `time_interval = s`.
You can choose whether only the last solution is kept (`only_last = true`, default), or all intermediate solutions (`only_last = false`). In the first case, the solution is saved as `[save_name].jls`. In the second case, `save_name` should contain a `#`, which is used as placeholder for the solution number.

@docs```
SaveSettings
```