#include <pcl/range_image/range_image_planar.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <chrono>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


Eigen::Affine3f random_pose() {
    srand(time(NULL));
    float angle_x = rand()*1.0/(RAND_MAX*1.0) * 2.0 * M_PI;
    float angle_y = rand()*1.0/(RAND_MAX*1.0) * 2.0 * M_PI;
    float angle_z = rand()*1.0/(RAND_MAX*1.0) * 2.0 * M_PI;

    Eigen::Affine3f R = Eigen::Affine3f::Identity();
    R.rotate(Eigen::AngleAxisf (angle_x, Eigen::Vector3f::UnitX()));
    R.rotate(Eigen::AngleAxisf (angle_y, Eigen::Vector3f::UnitY()));
    R.rotate(Eigen::AngleAxisf (angle_z, Eigen::Vector3f::UnitZ()));
    R.translation() <<  -R(0, 2), -R(1, 2), -R(2, 2);
    return R;

    // // float theta = M_PI/2;
    // Eigen::Affine3f R = Eigen::Affine3f::Identity();
    // R.translation() << 0.0, 0.0, -1.0;
    // // R.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitZ()));
    // return R;
}


Eigen::MatrixXf depth2pcd(Eigen::MatrixXf ranges, Eigen::Matrix3f intrinsics, Eigen::Matrix4f sensor_pose) {
    Eigen::Matrix3f inv_K = intrinsics.inverse();
    // Eigen::Matrix4f inv_P = sensor_pose.inverse();

    /* get depth value array */
    Eigen::MatrixXf depth(3, 0);
    for (int i = 0; i < ranges.rows(); i++) {
        for (int j = 0; j < ranges.cols(); j++) {
            if (isinf(-ranges(i, j)))
                continue;

            // float zc = ranges(i, j);
            float zc = intrinsics(0,0)*ranges(i, j)/sqrt(pow(intrinsics(1,2)-i, 2)+pow(intrinsics(0,2)-j, 2)+pow(intrinsics(0,0), 2));
            Eigen::Vector3f pixel(j*zc, i*zc, zc);
            depth.conservativeResize(depth.rows(), depth.cols() + 1);
            depth.col(depth.cols() - 1) = pixel;
        }
    }
    // std::cout << depth.cols() << "\n";

    /* image coordinates -> camera coordinates */
    Eigen::MatrixXf points1;
    points1 = inv_K * depth;
    // std::cout << points1.cols() << "\n";

    /* camera coordinates -> world coordinates */
    Eigen::MatrixXf vec_one = Eigen::MatrixXf::Ones(1, depth.cols());
    // std::cout << vec_one.cols() << "\n";
    Eigen::MatrixXf points2(4, depth.cols());
    points2 << points1,
               vec_one;
    // std::cout << points2.rows() << " " << points2.cols() << "\n";
    Eigen::MatrixXf points3 = (sensor_pose * points2).transpose().block(0, 0, depth.cols(), 3);
    // std::cout << points3.rows() << "\n";

    /* resample to 2048 */
    std::vector<int> idx;
    for (int i = 0; i < points3.rows(); i++) {
        idx.push_back(i);
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine eng(seed);
    std::shuffle(idx.begin(), idx.end(), eng);
    if (points3.rows() < 2048) {
        std::uniform_int_distribution<int> dist(0, points3.rows()-1);
        for (int i = 0; i < 2048-points3.rows(); i++) {
            idx.push_back(dist(eng));
        }
    }
    else {
        idx.resize(2048);
    }
    // Eigen::MatrixXf points4 = points3(idx, Eigen::all);
    Eigen::MatrixXf points4(0, 3);
    for (int i = 0; i < 2048; i++) {
        Eigen::Vector3f value = points3.row(idx[i]);
        points4.conservativeResize(points4.rows() + 1, points4.cols());
        points4.row(points4.rows() - 1) = value;
    }
    // std::cout << points4.rows() << "\n";
    return points4;
}


pcl::PointCloud<pcl::PointXYZ> numpy2pcd(pybind11::array_t<float> input) {
    // request a buffer descriptor from Python
    pybind11::buffer_info buffer_info = input.request();

    // extract data an shape of input array
    float *data = static_cast<float *>(buffer_info.ptr);
    std::vector<ssize_t> shape = buffer_info.shape;

    pcl::PointCloud<pcl::PointXYZ> pointCloud;
    for (int i = 0; i < shape[0]; i++) {
        pcl::PointXYZ point;
        point.x = data[i*shape[1] + 0];
        point.y = data[i*shape[1] + 1];
        point.z = data[i*shape[1] + 2];
        pointCloud.points.push_back(point);
    }
    pointCloud.width = pointCloud.size();
    pointCloud.height = 1;
    return pointCloud;
}


pybind11::array_t<float> eigen2numpy(Eigen::MatrixXf input) {
    // map to rowmajor storage
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> rmajor_input(input);

    std::vector<ssize_t> shape(2);
    shape[0] = input.rows();
    shape[1] = input.cols();
    return pybind11::array_t<float>(
        shape,  // shape
        {shape[1] * sizeof(float), sizeof(float)},  // strides
        rmajor_input.data());  // data pointer
}


pybind11::array_t<float> gen(pybind11::array_t<float> input) {
    pcl::PointCloud<pcl::PointXYZ> pointCloud1;
    pcl::PointCloud<pcl::PointXYZ> pointCloud2;
    
    // pcl::io::loadPCDFile("../data/1a04e3eab45ca15dd86060f189eb133.pcd", pointCloud1);
    pointCloud1 = numpy2pcd(input);
    // pcl::io::savePCDFile("1a04e3eab45ca15dd86060f189eb133.pcd", pointCloud1);
    // auto t1 = std::chrono::system_clock::now();

    int di_width = 160*10;
    int di_height = 120*10;
    float di_center_x = 80.0*10;
    float di_center_y = 60.0*10;
    float di_focal_length = 100.0*10;

    Eigen::Affine3f sensor_pose = random_pose();
    // Eigen::Affine3f sensor_pose = (Eigen::Affine3f) Eigen::Translation3f(0.0, 0.0, 0.0);
    // std::cout << sensor_pose.matrix() << "\n";
    pcl::RangeImagePlanar::CoordinateFrame coordinate_frame = pcl::RangeImagePlanar::CAMERA_FRAME;
    float noise_level = 0.0;
    float min_range = 0.0;

    pcl::RangeImagePlanar rangeImage;
    rangeImage.createFromPointCloudWithFixedSize(pointCloud1, di_width, di_height, di_center_x, di_center_y, 
                                    di_focal_length, di_focal_length, sensor_pose, coordinate_frame, noise_level, min_range);

    // std::cout << rangeImage << "\n";

    // auto t2 = std::chrono::system_clock::now();
    float* ranges = rangeImage.getRangesArray();
    Eigen::MatrixXf depth = Eigen::Map<Eigen::MatrixXf>(ranges, di_width, di_height).transpose();
    // std::cout << depth;
    unsigned char* rgb_image = pcl::visualization::FloatImageUtils::getVisualImage(ranges, rangeImage.width, rangeImage.height);
    pcl::io::saveRgbPNGFile("1a04e3eab45ca15dd86060f189eb133.png", rgb_image, rangeImage.width, rangeImage.height);

    // auto t3 = std::chrono::system_clock::now();
    Eigen::Matrix3f intrinsics; 
    intrinsics << di_focal_length, 0.0, di_center_x,
                  0.0, di_focal_length, di_center_y,
                  0.0, 0.0, 1.0;
    Eigen::MatrixXf output = depth2pcd(depth, intrinsics, sensor_pose.matrix());
    // auto t4 = std::chrono::system_clock::now();
    // std::cout << std::chrono::duration<float> (t2-t1).count() << "s\n";
    // std::cout << std::chrono::duration<float> (t3-t2).count() << "s\n";
    // std::cout << std::chrono::duration<float> (t4-t3).count() << "s\n";

    // for (int i = 0; i < output.rows(); i++) {
    //     pcl::PointXYZ point;
    //     point.x = output(i, 0);
    //     point.y = output(i, 1);
    //     point.z = output(i, 2);
    //     pointCloud2.points.push_back(point);
    // }
    // pointCloud2.width = pointCloud2.size();
    // pointCloud2.height = 1;
    // pcl::io::savePCDFile("1a04e3eab45ca15dd86060f189eb133.pcd", pointCloud2);
    // return 0;
    return eigen2numpy(output);
}


PYBIND11_MODULE(randpartial, m) {
    m.def("gen", &gen, "A function which generates random partial from the gt.");
}