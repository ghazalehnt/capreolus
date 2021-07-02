from os.path import join

from capreolus import ConfigOption, constants

from . import Collection

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class KITT(Collection):
    module_name = "kitt"
    _path = open(join(PACKAGE_PATH, "..", "paths_env_vars", "YGWYC_experiments_data_path"), 'r').read().strip() + "documents"
    collection_type = "TrecCollection"
    generator_type = "DefaultLuceneDocumentGenerator"
    config_spec = [ConfigOption("domain", "book")]


@Collection.register
class KITT_Inferred(Collection):
    module_name = "kitt_inferred"
    config_spec = [ConfigOption("domain", "book")]