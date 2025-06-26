from dataclasses import dataclass


@dataclass 
class KVTransferStats:
    """Container for transfer performance metrics"""
    transfer_durations: list[float]  # Transfer durations in seconds
    bytes_transferred: list[int]     # Bytes transferred per transfer
    num_blocks_transferred: list[int] # Number of blocks per transfer
    last_log_time: float             # Last time we logged metrics
    
    def __init__(self):
        self.reset(time.monotonic())
    
    def reset(self, now: float):
        self.transfer_durations = []
        self.bytes_transferred = []
        self.num_blocks_transferred = []
        self.last_log_time = now
    
    def record_transfer(self, duration: float, bytes_count: int, num_blocks: int):
        self.transfer_durations.append(duration)
        self.bytes_transferred.append(bytes_count)
        self.num_blocks_transferred.append(num_blocks)
    
    def get_throughput_stats(self, now: float) -> tuple[float, float, float]:
        """Get transfer throughput statistics"""
        time_elapsed = now - self.last_log_time
        if time_elapsed <= 0:
            return 0.0, 0.0, 0.0
            
        total_bytes = sum(self.bytes_transferred)
        total_blocks = sum(self.num_blocks_transferred)
        num_transfers = len(self.transfer_durations)
        
        # Bytes per second throughput
        bytes_per_sec = total_bytes / time_elapsed if time_elapsed > 0 else 0.0
        # Blocks per second throughput  
        blocks_per_sec = total_blocks / time_elapsed if time_elapsed > 0 else 0.0
        # Transfers per second
        transfers_per_sec = num_transfers / time_elapsed if time_elapsed > 0 else 0.0
        
        return bytes_per_sec, blocks_per_sec, transfers_per_sec
    
    def get_latency_stats(self) -> tuple[float, float, float]:
        """Get transfer latency statistics"""
        if not self.transfer_durations:
            return 0.0, 0.0, 0.0
            
        import numpy as np
        durations = np.array(self.transfer_durations)
        avg_latency = float(np.mean(durations))
        p50_latency = float(np.percentile(durations, 50))
        p95_latency = float(np.percentile(durations, 95))
        
        return avg_latency, p50_latency, p95_latency

class KVTransferLogging:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.transfer_durations = []
        self.bytes_transferred = []
        self.num_blocks_transferred = []

    def observe(self, transfer_stats: KVTransferStats):
        self.transfer_durations.append(transfer_stats.transfer_durations)
        self.bytes_transferred.append(transfer_stats.bytes_transferred)
        self.num_blocks_transferred.append(transfer_stats.num_blocks_transferred)
    
    def log(self):
        """Log transfer metrics periodically, similar to throughput logging"""
        # Only log if we have transfer data
        if self.transfer_metrics.transfer_durations:
            # Get throughput stats
            bytes_per_sec, blocks_per_sec, transfers_per_sec = \
                self.transfer_metrics.get_throughput_stats(now)
                
            # Get latency stats
            avg_latency, p50_latency, p95_latency = \
                self.transfer_metrics.get_latency_stats()
            
            # Format throughput for readability
            if bytes_per_sec >= 1024**3:  # GB/s
                bytes_throughput_str = f"{bytes_per_sec / (1024**3):.2f} GB/s"
            elif bytes_per_sec >= 1024**2:  # MB/s
                bytes_throughput_str = f"{bytes_per_sec / (1024**2):.1f} MB/s"
            elif bytes_per_sec >= 1024:  # KB/s
                bytes_throughput_str = f"{bytes_per_sec / 1024:.1f} KB/s"
            else:  # B/s
                bytes_throughput_str = f"{bytes_per_sec:.1f} B/s"
            
            # Log the metrics in a format similar to the existing throughput logs
            logger.info(
                "Engine %s: KV Transfer metrics: "
                "Avg transfer throughput: %s, "
                "Blocks/s: %.1f, Transfers/s: %.1f, "
                "Avg latency: %.3fs, P50: %.3fs, P95: %.3fs, "
                "Total transfers: %d",
                self.engine_id,
                bytes_throughput_str,
                blocks_per_sec,
                transfers_per_sec,
                avg_latency,
                p50_latency,
                p95_latency,
                len(self.transfer_metrics.transfer_durations)
            )
        else:
            # Log that no transfers occurred
            logger.debug(
                "Engine %s: No KV transfers in the last %.1fs",
                self.engine_id, time_since_last_log)
        
        # Reset metrics for next interval
        self.transfer_metrics.reset(now)