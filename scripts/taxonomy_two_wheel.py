"""Train-only supported two-wheel priors; fresh public initialization required.

The competition includes electric two-wheelers. Motorcycle and Scooter are
partial source priors, supported by the existing train-only geometry audit.
They are not an exact definition of the target class. Keep other groups fixed.
"""
import taxonomy_pooling

assert taxonomy_pooling.GROUPS[5] == [47]
taxonomy_pooling.GROUPS[5] = [47, 59, 130]
