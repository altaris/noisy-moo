"""
Noise wrappers.
* `nmoo.noises.gaussian.GaussianNoise`
* `nmoo.noises.uniform.UniformNoise`

Warning:

    In `pymoo`, constrained problems have a `G` component used to measure the
    validity of the inputs. Sometimes, that component depends on the actual
    value of the problem (i.e. the `F` component). See for instance [the
    implementation of
    CTP1](https://github.com/anyoptimization/pymoo/blob/7b719c330ff22c10980a21a272b3a047419279c8/pymoo/problems/multi/ctp.py#L84).
    Let $\\mathcal{P}$ be such a problem. Evaluating $\\mathcal{P}$ on an input
    $x$ gives the following output:
    $$
        \\mathcal{P} (x) = \\begin{cases}
            F = f (x)
            \\\\\\
            G = g (x, f (x))
        \\end{cases}
    $$
    where $f$ and $g$ are the non-noisy objective and constraint function,
    respectively. Now, let $\\mathcal{E}$ be a noise wrapper adding some random
    noise $\\varepsilon$ on the `F` component of $\\mathcal{P}$. Then we have
    $$
        \\mathcal{E} (\\mathcal{P} (x)) = \\begin{cases}
            F = f (x) + \\varepsilon
            \\\\\\
            G = g (x, f (x))
        \\end{cases}
    $$
    which is **NOT** the same as
    $$
        \\begin{cases}
            F = f (x) + \\varepsilon
            \\\\\\
            G = g (x, f (x) + \\varepsilon)
        \\end{cases}
    $$
    The latter may seem more natural, but breaks the wrapper hierarchy, as
    after getting the `F` value of $\\mathcal{P} (x)$, the noise layer
    $\\mathcal{E}$ would have to query $\\mathcal{P}$ again to calculate `G`
    while somehow substituting $f (x) + \\varepsilon$ for $f (x)$ in
    calculations.

"""
__docformat__ = "google"

from .gaussian import GaussianNoise
from .uniform import UniformNoise
