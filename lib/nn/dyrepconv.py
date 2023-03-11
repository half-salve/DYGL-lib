from .tgnconv import TGNConv
class DyRepConv(TGNConv):
    def __init__(self, ngh_finders, n_feat, e_feat,
                device, n_layers=2, n_heads=2, dropout=0.1, 
                use_memory=True, memory_update_at_start=True, 
                message_dimension=100, memory_dimension=500, 
                embedding_module_type="graph_attention", message_function="identity", 
                mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0, std_time_shift_dst=1, 
                n_neighbors=20, aggregator_type="last", memory_updater_type="rnn", 
                use_destination_embedding_in_message=True, use_source_embedding_in_message=False, 
                dyrep=True):
        super().__init__(ngh_finders, n_feat, e_feat, 
                        device, n_layers, n_heads, dropout, 
                        use_memory, memory_update_at_start, message_dimension, memory_dimension, 
                        embedding_module_type, message_function, 
                        mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst, 
                        n_neighbors, aggregator_type, memory_updater_type, 
                        use_destination_embedding_in_message, use_source_embedding_in_message, 
                        dyrep)
