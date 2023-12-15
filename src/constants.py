from pathlib import Path


COLORS = [(10, 118, 49), (35, 212, 23), (177, 212, 14), (212, 160, 26), (212, 88, 18), (125, 0, 27)]
COLORS = list(map(lambda x: (x[0] / 256, x[1] / 256, x[2] / 256), COLORS))

ROOT_DIR = Path(__file__).parent.parent

EXERCISE_NAMES_MAP = {
    "505886137": "Afscheidswoordje",  # https://dodona.be/nl/courses/1659/series/18384/activities/505886137/
    "933265977": "Symbolisch",  # https://dodona.be/nl/courses/1659/series/18384/activities/933265977/
    "1730686412": "Narcissuscodering",  # https://dodona.be/nl/courses/1659/series/18384/activities/1730686412/
    "1875043169": "Cocktailbar",  # https://dodona.be/nl/courses/1659/series/18385/activities/1875043169/
    "2046492002": "Antropomorfe emoji",  # https://dodona.be/nl/courses/1659/series/18385/activities/2046492002/
    "2146239081": "Kluizenaar",  # https://dodona.be/nl/courses/1659/series/18385/activities/2146239081/
}
