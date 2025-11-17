import os 

from anomalib.engine import Engine

# from custom.models import Detector

class Engine(Engine):
    def __init__(self, is_detector=False, *args, **kwargs):
        self.is_detector = is_detector
        self.detector = None
        self.results = None
        self.keeped_times = 0.0
        
        if not self.is_detector:
            super().__init__(*args, **kwargs)
        
    def fit(self, *args, **kwargs):
        if self.is_detector:
            pass
        else:
            super().fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        if self.is_detector:  
            self.detector = kwargs['model']
            self.datamodule = kwargs['datamodule']
            
            if os.listdir(self.datamodule.root) != ['ball']:
                print('Detector: ball 클래스가 없습니다..')
            else:
                results, keeped_times = self.detector.predict(source=os.path.join(self.datamodule.root, 'ball'))
                
                self.keeped_times = keeped_times
                self.results = results        
        else:
            return super().predict(*args, **kwargs)
        
    def report(self, *args, **kwargs):
        if self.is_detector:
            self.detector.report(self.results, self.keeped_times)   