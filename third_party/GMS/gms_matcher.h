#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <ctime>
#include <utility>
#include<numeric>
//using namespace std;
//using namespace cv;

#define THRESH_FACTOR 6

// 8 possible rotation and each one is 3 X 3 
const int mRotationPatterns[8][9] = {
	1,2,3,
	4,5,6,
	7,8,9,

	4,1,2,
	7,5,3,
	8,9,6,

	7,4,1,
	8,5,2,
	9,6,3,

	8,7,4,
	9,5,1,
	6,3,2,

	9,8,7,
	6,5,4,
	3,2,1,

	6,9,8,
	3,5,7,
	2,1,4,

	3,6,9,
	2,5,8,
	1,4,7,

	2,3,6,
	1,5,9,
	4,7,8
};

// 5 level scales
const double mScaleRatios[5] = { 1.0, 1.0 / 2, 1.0 / sqrt(2.0), sqrt(2.0), 2.0 };


class gms_matcher
{
public:
	// OpenCV Keypoints & Correspond Image Size & Nearest Neighbor Matches 
        gms_matcher(const std::vector<cv::KeyPoint> &vkp1,
                    const cv::Size size1,
                    const std::vector<cv::KeyPoint> &vkp2,
                    const cv::Size size2,
                    const std::vector<cv::DMatch> &vDMatches)
	{
		// Input initialize
		NormalizePoints(vkp1, size1, mvP1);
		NormalizePoints(vkp2, size2, mvP2);
		mNumberMatches = vDMatches.size();
		ConvertMatches(vDMatches, mvMatches);

		// Grid initialize
                mGridSizeLeft = cv::Size(20, 20);
		mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;

		// Initialize the neihbor of left grid 
                mGridNeighborLeft = cv::Mat::zeros(mGridNumberLeft, 9, CV_32SC1);
		InitalizeNiehbors(mGridNeighborLeft, mGridSizeLeft);
        }
        gms_matcher(const std::vector<cv::Point2f> &vkp1,
                    const cv::Size size1,
                    const std::vector<cv::Point2f> &vkp2,
                    const cv::Size size2,
                    const std::vector<cv::DMatch> &vDMatches)
        {
                // Input initialize
                NormalizePoints2f(vkp1, size1, mvP1);
                NormalizePoints2f(vkp2, size2, mvP2);
                mNumberMatches = vDMatches.size();
                ConvertMatches(vDMatches, mvMatches);

                // Grid initialize
                mGridSizeLeft = cv::Size(20, 20);
                mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;

                // Initialize the neihbor of left grid
                mGridNeighborLeft = cv::Mat::zeros(mGridNumberLeft, 9, CV_32SC1);
                InitalizeNiehbors(mGridNeighborLeft, mGridSizeLeft);
        }
        ~gms_matcher() {}

private:

	// Normalized Points
        std::vector<cv::Point2f> mvP1, mvP2;

	// Matches
        std::vector<std::pair<int, int> > mvMatches;

	// Number of Matches
	size_t mNumberMatches;

	// Grid Size
        cv::Size mGridSizeLeft, mGridSizeRight;
	int mGridNumberLeft;
	int mGridNumberRight;

	// x	  : left grid idx
	// y      :  right grid idx
	// value  : how many matches from idx_left to idx_right
        cv::Mat mMotionStatistics;

	// 
        std::vector<int> mNumberPointsInPerCellLeft;

	// Inldex  : grid_idx_left
	// Value   : grid_idx_right
        std::vector<int> mCellPairs;

	// Every Matches has a cell-pair 
	// first  : grid_idx_left
	// second : grid_idx_right
        std::vector<std::pair<int, int> > mvMatchPairs;

	// Inlier Mask for output
        std::vector<bool> mvbInlierMask;

	//
        cv::Mat mGridNeighborLeft;
        cv::Mat mGridNeighborRight;

public:

	// Get Inlier Mask
	// Return number of inliers 
        int GetInlierMask(std::vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false);

private:

	// Normalize Key Points to Range(0 - 1)
        void NormalizePoints(const std::vector<cv::KeyPoint> &kp, const cv::Size &size, std::vector<cv::Point2f> &npts) {
		const size_t numP = kp.size();
		const int width   = size.width;
		const int height  = size.height;
		npts.resize(numP);

		for (size_t i = 0; i < numP; i++)
		{
			npts[i].x = kp[i].pt.x / width;
			npts[i].y = kp[i].pt.y / height;
		}
	}

        void NormalizePoints2f(const std::vector<cv::Point2f> &kp, const cv::Size &size, std::vector<cv::Point2f> &npts) {
                const size_t numP = kp.size();
                const int width   = size.width;
                const int height  = size.height;
                npts.resize(numP);

                for (size_t i = 0; i < numP; i++)
                {
                        npts[i].x = kp[i].x / width;
                        npts[i].y = kp[i].y / height;
                }
        }

	// Convert OpenCV DMatch to Match (pair<int, int>)
        void ConvertMatches(const std::vector<cv::DMatch> &vDMatches, std::vector<std::pair<int, int> > &vMatches) {
		vMatches.resize(mNumberMatches);
		for (size_t i = 0; i < mNumberMatches; i++)
		{
                        vMatches[i] = std::pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
		}
	}

        int GetGridIndexLeft(const cv::Point2f &pt, int type) {
		int x = 0, y = 0;

		if (type == 1) {
                        x = std::floor(pt.x * mGridSizeLeft.width);
                        y = std::floor(pt.y * mGridSizeLeft.height);

			if (y >= mGridSizeLeft.height || x >= mGridSizeLeft.width){
				return -1;
			}
		}

		if (type == 2) {
                        x = std::floor(pt.x * mGridSizeLeft.width + 0.5);
                        y = std::floor(pt.y * mGridSizeLeft.height);

			if (x >= mGridSizeLeft.width || x < 1) {
				return -1;
			}
		}

		if (type == 3) {
                        x = std::floor(pt.x * mGridSizeLeft.width);
                        y = std::floor(pt.y * mGridSizeLeft.height + 0.5);

			if (y >= mGridSizeLeft.height || y < 1) {
				return -1;
			}
		}

		if (type == 4) {
                        x = std::floor(pt.x * mGridSizeLeft.width + 0.5);
                        y = std::floor(pt.y * mGridSizeLeft.height + 0.5);

			if (y >= mGridSizeLeft.height || y < 1 || x >= mGridSizeLeft.width || x < 1) {
				return -1;
			}
		}

		return x + y * mGridSizeLeft.width;
	}

        int GetGridIndexRight(const cv::Point2f &pt) {
                int x = std::floor(pt.x * mGridSizeRight.width);
                int y = std::floor(pt.y * mGridSizeRight.height);

		return x + y * mGridSizeRight.width;
	}

	// Assign Matches to Cell Pairs 
	void AssignMatchPairs(int GridType);

	// Verify Cell Pairs
	void VerifyCellPairs(int RotationType);

	// Get Neighbor 9
        std::vector<int> GetNB9(const int idx, const cv::Size& GridSize) {
                std::vector<int> NB9(9, -1);

		int idx_x = idx % GridSize.width;
		int idx_y = idx / GridSize.width;

		for (int yi = -1; yi <= 1; yi++)
		{
			for (int xi = -1; xi <= 1; xi++)
			{	
				int idx_xx = idx_x + xi;
				int idx_yy = idx_y + yi;

				if (idx_xx < 0 || idx_xx >= GridSize.width || idx_yy < 0 || idx_yy >= GridSize.height)
					continue;

				NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * GridSize.width;
			}
		}
		return NB9;
	}

        void InitalizeNiehbors(cv::Mat &neighbor, const cv::Size& GridSize) {
		for (int i = 0; i < neighbor.rows; i++)
		{
                        std::vector<int> NB9 = GetNB9(i, GridSize);
			int *data = neighbor.ptr<int>(i);
			memcpy(data, &NB9[0], sizeof(int) * 9);
		}
	}

	void SetScale(int Scale) {
		// Set Scale
		mGridSizeRight.width = mGridSizeLeft.width  * mScaleRatios[Scale];
		mGridSizeRight.height = mGridSizeLeft.height * mScaleRatios[Scale];
		mGridNumberRight = mGridSizeRight.width * mGridSizeRight.height;

		// Initialize the neihbor of right grid 
                mGridNeighborRight = cv::Mat::zeros(mGridNumberRight, 9, CV_32SC1);
		InitalizeNiehbors(mGridNeighborRight, mGridSizeRight);
	}

	// Run 
	int run(int RotationType);
};
