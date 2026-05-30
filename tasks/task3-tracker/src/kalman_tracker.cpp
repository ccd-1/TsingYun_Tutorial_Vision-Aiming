#include "kalman_tracker.hpp"

#include <stdexcept>

namespace hw
{
    KalmanTracker::KalmanTracker() = default;

    bool KalmanTracker::isTracking() const
    {
        return tracking_;
    }

    void KalmanTracker::reset()
    {
        tracking_ = false;
        x_ = AxisFilter{};
        y_ = AxisFilter{};
        z_ = AxisFilter{};
    }

    void KalmanTracker::AxisFilter::reset(double measured_position)
    {
        position = measured_position;
        velocity = 0.0;
        p00 = 1.0;
        p01 = 0.0;
        p10 = 0.0;
        p11 = 1.0;
    }

    void KalmanTracker::AxisFilter::predict(double dt, double process_noise)
    {
        dt = dt < 0.0 ? 0.0 : dt;
        position += velocity * dt;

        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt2 * dt2;

        const double q00 = process_noise * (dt4 / 4.0);
        const double q01 = process_noise * (dt3 / 2.0);
        const double q10 = q01;
        const double q11 = process_noise * dt2;

        const double new_p00 = p00 + p01 * dt + p10 * dt + p11 * dt2 + q00;
        const double new_p01 = p01 + p11 * dt + q01;
        const double new_p10 = p10 + p11 * dt + q10;
        const double new_p11 = p11 + q11;

        p00 = new_p00;
        p01 = new_p01;
        p10 = new_p10;
        p11 = new_p11;
    }

    void KalmanTracker::AxisFilter::update(double measured_position, double measurement_noise)
    {
        const double residual = measured_position - position;
        const double s = p00 + measurement_noise;
        if (s <= 0.0)
        {
            return;
        }

        const double k0 = p00 / s;
        const double k1 = p10 / s;

        position += k0 * residual;
        velocity += k1 * residual;

        const double new_p00 = p00 - k0 * p00;
        const double new_p01 = p01 - k0 * p01;
        const double new_p10 = p10 - k1 * p00;
        const double new_p11 = p11 - k1 * p01;

        p00 = new_p00;
        p01 = new_p01;
        p10 = new_p10;
        p11 = new_p11;
    }

    TrackState KalmanTracker::update(const Vec3 &measurement, double dt)
    {
        if (!tracking_)
        {
            x_.reset(measurement.x);
            y_.reset(measurement.y);
            z_.reset(measurement.z);
            tracking_ = true;
            return stateFromFilters();
        }

        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);

        x_.update(measurement.x, measurement_noise_);
        y_.update(measurement.y, measurement_noise_);
        z_.update(measurement.z, measurement_noise_);

        return stateFromFilters();
    }

    TrackState KalmanTracker::predict(double dt)
    {
        if (!tracking_)
        {
            return TrackState{};
        }

        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);

        return stateFromFilters();
    }

    TrackState KalmanTracker::stateFromFilters() const
    {
        return {
            true,
            {x_.position, y_.position, z_.position},
            {x_.velocity, y_.velocity, z_.velocity},
        };
    }
} // namespace hw
