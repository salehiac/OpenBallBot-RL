from gymnasium.envs.registration import register
import importlib.resources

with importlib.resources.path("ballbotgym.assets", "bbot.xml") as path:
    _xml_path=str(path)

register(
    id="ballbot-v0.1",
    entry_point="ballbotgym.bbot_env:BBotSimulation",
    kwargs={
        "xml_path":_xml_path
        }
)

