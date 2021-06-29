from os.path import join
from pathlib import Path

from capreolus import ConfigOption, Dependency, constants

from . import Benchmark
from ..utils.trec import load_trec_topics

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class KITT(Benchmark):
    """KITT(YGWYC) dataset benchmark"""

    module_name = "kitt"
    dependencies = [Dependency(key="collection", module="collection", name="kitt")]
    config_spec = [ConfigOption("query_type", "query"),
                   ConfigOption("domain", "book"),
                   ConfigOption("assessed_set", "all")]
    DATA_DIR = Path(open(join(PACKAGE_PATH, "..", "paths_env_vars", "YGWYC_experiments_data_path"), 'r').read().strip())

    def build(self):
        self.qrel_file = self.DATA_DIR / "judgements" / "{}_judgements_{}".format(self.config["domain"], self.config["assessed_set"])
        self.topic_file = self.DATA_DIR / "topics" / f"{self.domain}_topics.{self.query_type}.txt"
        self.fold_file = self.DATA_DIR / "splits" / "{}_folds.json".format(self.config["domain"])

    @property
    def topics(self):
        if not hasattr(self, "_topics"):
            self._topics = load_trec_topics(self.topic_file)
            assert self.query_type not in self._topics
            self._topics[self.query_type] = self._topics["title"]
        return self._topics

    @property
    def query_type(self):
        return self.config["query_type"]

    @property
    def domain(self):
        return self.config["domain"]
