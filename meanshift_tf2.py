import tensorflow as tf
import numpy as np

def main():
    ms = MeanShiftTF()
    n_points = 50
    n_features = 6
    np.random.seed(123)
    data = np.random.random((n_points, n_features)).astype(np.float32)
    peaks, labels = ms.apply(data, radius=0.1, chunk_size=4)
    print(peaks)
    print(labels)


class MeanShiftTF:
    def __init__(self):
        pass
        # self.radius_t = tf.Tensor(shape=(), dtype=tf.float32)

    def mean_shift_step(self, data_t, max_shift_t, is_near_path_t):
        # data batch data_t has dims [4,6]
        data1_t = tf.expand_dims(data_t, 2) # [4, 6] -> [4, 6, 1]
        # all data self.orig_data_all has dims [50, 6]
        data2_t = tf.transpose(tf.expand_dims(self.orig_data_all_t, 2), [2, 1, 0]) # [50, 6] -> [50, 6, 1] -> [1, 6, 50]

        # we compute the distance of every batch data point (shape [4,6,1]) in every feature axis ([6])
        # from all the dataset points (shape [1,6,50]), the distance matrix is [4,6,50]
        # we take a normalization along the feature axis to obtain a 1d distance matrix of every batch poiunt to other dataset points [4,1,50]
        dist_t = tf.norm(data1_t - data2_t, axis=1, keepdims=True) # [4,6,1] - [1,6,50] = [4,6,50] -> norm -> [4,1,50]

        # we check if any distance between points falls under radius_t, the bool tensor (casted as float) is [4,1,50]
        # note: diagonal elements are surely 1. at first pass (distance of batch points from themselves is 0.)
        # we add a zeros_like(data2_t) [1,6,50] to expand on the feature dim
        is_within_radius_t = tf.cast(dist_t <= self.radius_t, tf.float32) + tf.zeros_like(data2_t)# [4,1,50] -> [4,6,50]

        # k is the number of points closer than radius_t to other points in the dataset
        num_within_radius_t = tf.math.count_nonzero(is_within_radius_t, axis=2, dtype=data_t.dtype)# [4,6,50] -> [k, 6]
        
        # is_within_radius_t is a proximity matrix (1 if closer than radius_t, 0 otherwise)
        # data_within_radius_t is just masked close points
        data_within_radius_t = is_within_radius_t * data2_t # [k,6] -> [k,6,50]
        sum_within_radius_t = tf.reduce_sum(data_within_radius_t, axis=2)# [k,6,50] -> [k,6]
        shifted_data_t = sum_within_radius_t / num_within_radius_t # [k,6] -> [k,6]

        shift_t = tf.norm(shifted_data_t - data_t, axis=1)# [k,6] - [4,6] -> k
        max_shift_t = tf.reduce_max(shift_t)# k -> 1

        #[4, 1, 50] -> [4,50]
        is_near_path_new_t = tf.logical_or(tf.squeeze(dist_t, 1) <= self.radius_t/2, is_near_path_t)
        return shifted_data_t, max_shift_t, is_near_path_new_t
    
    
    def apply(self, data, radius, chunk_size=4):
        labels = np.full(len(data), fill_value=-1, dtype=int)
        indices = np.argwhere(labels==-1)[:chunk_size, 0]
        peaks = np.empty_like(data)
        thres_sq = (radius/2) ** 2
        n_peaks = 0

        self.radius_t = tf.convert_to_tensor(radius)
        max_shift_t = tf.convert_to_tensor(np.float32(np.inf))
        self.orig_data_all_t = tf.convert_to_tensor(data, dtype=tf.float32)

                                            
        while np.any(labels==-1):
            indices = np.argwhere(labels==-1)[:chunk_size, 0]
            indices_t = tf.convert_to_tensor(indices)
            data_batch_t = tf.gather(self.orig_data_all_t, indices_t)
            is_near_path_t = tf.zeros(shape=(tf.shape(data_batch_t)[0],
                        tf.shape(self.orig_data_all_t)[0]),
                        dtype=tf.bool)

            while max_shift_t > 0.1: # convergence
                data_batch_t, max_shift_t, is_near_path_t = self.mean_shift_step(
                    data_batch_t, max_shift_t, is_near_path_t
                )

            for idx, peak, is_near_this_path in zip(indices, data_batch_t, is_near_path_t):
                label = None

                if n_peaks > 0:
                    dist = np.linalg.norm(peaks[:n_peaks] - peak, axis=1)
                    label_of_nearest_peak = np.argmin(dist)

                    if dist[label_of_nearest_peak] <= radius/2:
                        label = label_of_nearest_peak
                
                if label is None:
                    label = n_peaks
                    peaks[n_peaks] = peak
                    n_peaks += 1

                    dist_sq = np.linalg.norm(data - peak, axis=1)
                    labels[dist_sq <= radius] = label

                labels[is_near_this_path.numpy()] = label
        return peaks[:n_peaks], labels


        
if __name__ == "__main__":
    main()