from os.path import join
from pathlib import Path

from capreolus import ConfigOption, Dependency, constants

from . import Benchmark
from ..utils.trec import load_trec_topics

PACKAGE_PATH = constants["PACKAGE_PATH"]

class KITT(Benchmark):
    """KITT(YGWYC) dataset benchmark"""

    module_name = "kitt"
    dependencies = [Dependency(key="collection", module="collection", name="kitt")]
    config_spec = [ConfigOption("query_type", "query"),
                   ConfigOption("domain", "book"),
                   ConfigOption("assessed_set", "all")]
    DATA_DIR = Path(open(join(PACKAGE_PATH, "..", "paths_env_vars", "YGWYC_experiments_data_path"), 'r').read().strip())

    @property
    def qrel_file(self):
        fn = "{}_judgements_{}".format(self.config["domain"], self.config["assessed_set"])
        return self.DATA_DIR / "judgements" / fn

    @property
    def fold_file(self):
        fn = "{}_folds.json".format(self.config["domain"])
        return self.DATA_DIR / "splits" / fn

    @property
    def topic_file(self):
        fn = f"{self.domain}_topics.{self.query_type}.txt"
        return self.DATA_DIR / "topics" / fn

    @property
    def topics(self):
        if not hasattr(self, "_topics"):
            self._topics = load_trec_topics(self.topic_file)
            assert self.query_type not in self._topics
            self._topics[self.query_type] = self._topics["title"]
        return self._topics

    @property
    def query_type(self):
        return self.cfg["querytype"]

    @property
    def domain(self):
        return self.cfg["domain"]
