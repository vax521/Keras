import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
def opitimize_cpu():
    # 充分使用CPU
    config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,
                            allow_soft_placement=True, device_count={'CPU': 4})
    session = tf.Session(config=config)
    KTF.set_session(session)