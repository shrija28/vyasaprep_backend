# Vercel Python serverless entry point.
#
# Vercel's Python runtime expects a WSGI-callable named ``app`` (or
# ``handler``) at the module level.  We simply re-export the Flask
# application object that ``smartkcet.main`` already constructs so that
# no second Flask instance is created.
#
# Vercel routes every inbound request through this file (see vercel.json).

from smartkcet.main import app  # noqa: F401  – re-exported as WSGI app
