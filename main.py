from inltk.inltk import setup
import asyncio

if asyncio.get_event_loop().is_running():
    print("An event loop is already running.")
else:
    setup('kn')