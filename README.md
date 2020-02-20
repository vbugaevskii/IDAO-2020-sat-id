# IDAO-2020-sat-id

This repository is the solution for the Online Round of [IDAO-2020](https://idao.world/) ML-contest of team **A Jelly Fish Swam In The Sea**.

Team members:
* [Eugen Bobrov](https://github.com/eugenbobrov) (leader);
* [Vladimir Bugaevskii](https://github.com/vbugaevskii);
* [Andrew Soroka](https://github.com/ASorok).

Baseline solution and data from contest's organizers can be found [here](https://yadi.sk/d/0zYx00gSraxZ3w).

The optimized metric is [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error). SMAPE is averaged by all targets (`x`, `y`, `z`, `Vx`, `Vy`, `Vz`) and by all satellites' points.

&nbsp;
![smape.svg](smape.svg)
&nbsp;

**NOTE:** Final score on leaderborder is calculated as `100 * (1 - MEAN(SMAPE))`.

## Track 1

The original simulated coordinates and velocities turned out to be inaccurate, so we were not able to use them in ML-model. Instead we use [Exponential Smoothing model](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html) to predict simulated coordinates and velocities more accurately, and then new predicted values are used instead of simulated ones.

[`Track1-ExponentialSmoothing.ipynb`](Track1-ExponentialSmoothing.ipynb) contains models for predciting new simulated coordinates and velocities.

[`Track1-LightGBM.ipynb`](Track1-LightGBM.ipynb) contains LightGBM models that are used to make simulated values more precise.

## Track 2

The solution uses only exponential smoothing models from Track A, see [`Track1-ExponentialSmoothing.ipynb`](Track1-ExponentialSmoothing.ipynb).

[`Track2`](Track2) contains python code for submission.

## Leaderboard

|             | public | private |
|-------------|--------|---------|
| **Track 1** | 87.07  | 89.42   |
| **Track 2** | 86.58  | 86.63   |

Our team took the 32-th place on the private leaderboard according to both tracks.
