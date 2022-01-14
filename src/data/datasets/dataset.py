
class Dataset:

    @staticmethod
    def load_instances(self, cfg, dirname: str, split: str, **kwargs):
        """
        Load the full (or part of the dataset)
        """        
        raise NotImplementedError

    def generate_mappings(self, cfg, **kwargs):
        """
        Generate mappings sub2super and super2sub
        """
        raise NotImplementedError
