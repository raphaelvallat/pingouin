# Import pingouin objects
from .utils import *
from .bayesian import *
from .datasets import *
from .distribution import *
from .effsize import *
from .equivalence import *
from .multicomp import *
from .multivariate import *
from .parametric import *
from .nonparametric import *
from .correlation import *
from .circular import *
from .pairwise import *
from .power import *
from .reliability import *
from .regression import *
from .plotting import *
from .contingency import *
from .config import *

# Current version
__version__ = "0.5.2"

# Warn if a newer version of Pingouin is available
from outdated import warn_if_outdated

warn_if_outdated("pingouin", __version__)

# load default options
set_default_options()
