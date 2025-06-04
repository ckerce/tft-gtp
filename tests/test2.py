import sys
sys.path.insert(0, 'src')
from utils.json_logger import MetricsAggregator

agg = MetricsAggregator()
agg.update(loss=2.5, accuracy=0.8)
result = agg.get_averages()
print("✅ Works:", result)