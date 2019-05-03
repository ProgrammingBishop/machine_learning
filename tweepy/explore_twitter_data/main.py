# Imports
# ==================================================
from DefineApplication import DefineApplication
from RunApplication    import RunApplication

import configurations as c


# Application
# ==================================================
if __name__ == "__main__":
    # PRIVATE
    define_app = DefineApplication()
    steps      = define_app.define_program()

    RunApplication( steps, c )