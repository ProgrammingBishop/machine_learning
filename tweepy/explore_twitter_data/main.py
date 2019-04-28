# Imports
# ==================================================
import configurations as c

from DefineApplication import DefineApplication
from RunApplication    import RunApplication


if __name__ == "__main__":
    define_app = DefineApplication()
    steps      = define_app.define_program()

    RunApplication( steps, c )