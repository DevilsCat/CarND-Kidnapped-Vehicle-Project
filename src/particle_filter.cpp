/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random>
#include <map>

#include "helper_functions.h"
#include "particle_filter.h"

using namespace std;

#define STD_X_INDEX 0
#define STD_Y_INDEX 1
#define STD_THETA_INDEX 2

#define YAW_RATE_THRESHOLD 0.0001

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles. Initialize all particles to first
  // position (based on estimates of x, y, theta and their uncertainties 
  // from GPS) and all weights to 1.  Add random Gaussian noise to each 
  // particle.
  // NOTE: Consult particle_filter.h for more information about this method 
  // (and others in this file).
  num_particles = 10;

  // Creates normal distribution for x, y and theta.
  normal_distribution<double> dist_x(x, std[STD_X_INDEX]);
  normal_distribution<double> dist_y(y, std[STD_Y_INDEX]);
  normal_distribution<double> dist_theta(theta, std[STD_THETA_INDEX]);

  for (int i = 0; i < num_particles; ++i) {
    Particle particle;

    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;

    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  
  for (Particle& particle : particles) {
    // Add measurements
    if (fabs(yaw_rate) < YAW_RATE_THRESHOLD) {
      // Use constant yaw speed model
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    } else {
      const double velocity_yaw_rate = velocity / yaw_rate;
      const double yaw_rate_delta = yaw_rate * delta_t;
      const double post_theta = particle.theta + yaw_rate_delta;

      particle.x += 
        velocity_yaw_rate * (sin(post_theta) - sin(particle.theta));
      particle.y += 
        velocity_yaw_rate * (cos(particle.theta) - cos(post_theta));
      particle.theta += yaw_rate_delta;
    }

    // Add random Gaussian noise
    normal_distribution<double> dist_x(particle.x, std_pos[STD_X_INDEX]);
    normal_distribution<double> dist_y(particle.y, std_pos[STD_Y_INDEX]);
    normal_distribution<double> dist_theta(
        particle.theta, std_pos[STD_THETA_INDEX]);

    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.
  for (LandmarkObs& obs : observations) {
    double min_dist = numeric_limits<double>::max();

    for (LandmarkObs& predictedObs : predicted) {
      double dist_between = dist(obs.x, obs.y, predictedObs.x, predictedObs.y);

      if (min_dist > dist_between) {
        min_dist = dist_between;
        obs.id = predictedObs.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  for (Particle& particle : particles) {
    const double p_x = particle.x;
    const double p_y = particle.y;
    const double p_theta = particle.theta;

    // Filter a list of landmarks within the sensor_range and transform them
    // into LandmarkObs objects (in map coordinate).
    vector<LandmarkObs> predicted;
    for (const auto& landmark : map_landmarks.landmark_list) {
      const float l_x = landmark.x_f;
      const float l_y = landmark.y_f;

      if (fabs(p_x - l_x) > sensor_range || fabs(p_y - l_y) > sensor_range) {
        continue;
      }

      LandmarkObs predictedObs;
      predictedObs.id = landmark.id_i;
      predictedObs.x = l_x;
      predictedObs.y = l_y;
      predicted.push_back(predictedObs);
    }

    // Transform the observations from particle coordinate to map coordinate.
    vector<LandmarkObs> transformed_observations;
    transform(
        observations.begin(), 
        observations.end(), 
        back_inserter(transformed_observations), 
        [&p_x, &p_y, &p_theta](const LandmarkObs& obs) {
          LandmarkObs transformed_obs;
          transformed_obs.id = obs.id;
          transformed_obs.x = 
              obs.x * cos(p_theta) - obs.y * sin(p_theta) + p_x;
          transformed_obs.y = 
              obs.x * sin(p_theta) + obs.y * cos(p_theta) + p_y;

          return transformed_obs; 
        });

    // Perform data associations
    dataAssociation(predicted, transformed_observations);
    
    // Make a map mapping the id --> map observation for fast access.
    map<int, LandmarkObs> predictedMap;
    for (auto it = predicted.begin(); it != predicted.end(); ++it) {
      predictedMap[it->id] = *it;
    }

    // Calculate weight for the particle using each observation (in map 
    // coordinate).
    particle.weight = 1.0;

    for (const LandmarkObs& obs : transformed_observations) {
      const double o_x = obs.x;
      const double o_y = obs.y;
      const double u_x = predictedMap[obs.id].x;
      const double u_y = predictedMap[obs.id].y;
      const double std_x = std_landmark[0];
      const double std_y = std_landmark[1];

      particle.weight *= 
        1.0 / (2 * M_PI * std_x * std_y)
          * exp(
            - (o_x-u_x)*(o_x-u_x) / (2*std_x*std_x) 
            - (o_y-u_y)*(o_y-u_y) / (2*std_y*std_y));
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional
  // to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<Particle> new_particles;

  vector<double> weights;
  transform(
    particles.begin(), 
    particles.end(), 
    back_inserter(weights), 
    [](const Particle& p) { return p.weight; });

  discrete_distribution<int> d(weights.begin(), weights.end());
  for (int i = 0; i < num_particles; ++i) {
    new_particles.push_back(particles[d(gen)]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
