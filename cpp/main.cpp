#include <iostream>
#include <vector>
#include <math.h>
#include <numeric>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_multimin.h>
#include <algorithm>
// #include <chrono>

double lambda = 0;

double mean(std::vector<double> v){
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean_v = sum / v.size();
    return mean_v;
}

double std_dev(std::vector<double> v, double mean){
    double stdev = 0;
    for(int i=0; i<v.size(); i++){
        stdev += pow(v[i] - mean, 2);
    }
    stdev = sqrt(stdev/v.size());

    return stdev;
}

std::vector<double> sub_pow(std::vector<double> x, std::vector<double> y, int power){
    std::vector<double> res;
    for(int i=0; i<x.size(); i++){
        res.push_back(pow(x[i] - y[i], power));
    }
    return res;
}

double cost_f (const gsl_vector *v, void *params)
{
    int length = int(*(double*)params);

    double x0, x1, x2, x3, y0, y1, y2, y3;
    x0 = gsl_vector_get(v, 0);
    x1 = gsl_vector_get(v, 1);
    x2 = gsl_vector_get(v, 2);
    x3 = gsl_vector_get(v, 3);
    y0 = gsl_vector_get(v, 4);
    y1 = gsl_vector_get(v, 5);
    y2 = gsl_vector_get(v, 6);
    y3 = gsl_vector_get(v, 7);

    double *time_pointer = (double *)params;
    std::vector<double> res;

    for(int i =1 ; i<length+1; i++){
        res.push_back(x0 + x1 * *(time_pointer + i) + x2 * pow(*(time_pointer + i), 2) + x3 * pow(*(time_pointer + i), 3));
    }
    for(int i =1 ; i<length+1; i++){
        res.push_back(y0 + y1 * *(time_pointer + i) + y2 * pow(*(time_pointer + i), 2) + y3 * pow(*(time_pointer + i), 3));
    }

    double reg = 0;
    // reg += pow(x0, 2);
    reg += pow(x1, 2);
    reg += pow(x2, 2);
    reg += pow(x3, 2);
    // reg += pow(y0, 2);
    reg += pow(y1, 2);
    reg += pow(y2, 2);
    reg += pow(y3, 2);

    double reg_weight = lambda;

    std::vector<double> Y;
    for(int i =length+1 ; i<length*3+1; i++){
        Y.push_back(*(time_pointer + i));
    }

    return mean(sub_pow(res, Y, 2)) + 0.5 * reg * reg_weight;
}

double calculate_grad(std::vector<double> theta_df_res, std::vector<double> res, std::vector<double> Y, double reg, double reg_weight){
    std::vector<double> grad(Y.size());
    for(int i=0; i<Y.size(); i++){
        grad[i] = (res[i] - Y[i]) * theta_df_res[i] + reg * reg_weight;
    }

    return 2*mean(grad);
}

/* The gradient of f, df = (df/dx, df/dy). */
void gradient (const gsl_vector *v, void *params,
       gsl_vector *df)
{
    int length = int(*(double*)params);

    double x0, x1, x2, x3, y0, y1, y2, y3;
    x0 = gsl_vector_get(v, 0);
    x1 = gsl_vector_get(v, 1);
    x2 = gsl_vector_get(v, 2);
    x3 = gsl_vector_get(v, 3);
    y0 = gsl_vector_get(v, 4);
    y1 = gsl_vector_get(v, 5);
    y2 = gsl_vector_get(v, 6);
    y3 = gsl_vector_get(v, 7);

    double *time_pointer = (double *)params;
    std::vector<double> res_x;
    std::vector<double> res_y;
    std::vector<std::vector<double>> theta_df_res(8);

    for(int i=1; i<length+1; i++){
        res_x.push_back(x0 + x1 * *(time_pointer + i) + x2 * pow(*(time_pointer + i), 2) + x3 * pow(*(time_pointer + i), 3));
        theta_df_res[0].push_back(1);
        theta_df_res[1].push_back(*(time_pointer + i));
        theta_df_res[2].push_back(pow(*(time_pointer + i), 2));
        theta_df_res[3].push_back(pow(*(time_pointer + i), 3));

        res_y.push_back(y0 + y1 * *(time_pointer + i) + y2 * pow(*(time_pointer + i), 2) + y3 * pow(*(time_pointer + i), 3));
        theta_df_res[4].push_back(1);
        theta_df_res[5].push_back(*(time_pointer + i));
        theta_df_res[6].push_back(pow(*(time_pointer + i), 2));
        theta_df_res[7].push_back(pow(*(time_pointer + i), 3));
    }

    double reg_weight = lambda;
    double reg[8] = {0, x1, x2, x3, 0, y1, y2, y3};

    std::vector<double> Y_x;
    std::vector<double> Y_y;
    for(int i=length+1; i<length*2+1; i++){
        Y_x.push_back(*(time_pointer + i));
    }
    for(int i=length*2+1; i<length*3+1; i++){
        Y_y.push_back(*(time_pointer + i));
    }

    for(int i = 0; i < 8; i++){
        if (i < 4)
            gsl_vector_set(df, i, calculate_grad(theta_df_res[i], res_x, Y_x, reg[i], reg_weight));
        else
            gsl_vector_set(df, i, calculate_grad(theta_df_res[i], res_y, Y_y, reg[i], reg_weight));
    }
}

/* Compute both f and df together. */
void grad_cost (const gsl_vector *x, void *params,
        double *f, gsl_vector *df)
{
  *f = cost_f(x, params);
  gradient(x, params, df);
}

// double* trajectory_interpolator2_cpp(double *_data, int data_length)
extern "C" double* trajectory_interpolator2_cpp(double *_data, int data_length, double _lambda)
{
    lambda = _lambda;
    int length = data_length / 3;

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> tstamps;

    for(int i=0; i<length ; i++){
        tstamps.push_back(_data[i]);
    }
    for(int i=length; i<length*2 ; i++){
        x.push_back(_data[i]);
    }
    for(int i=length*2; i<length*3 ; i++){
        y.push_back(_data[i]);
    }

    double mean_x = mean(x);
    double mean_y = mean(y);
    double std_x = std_dev(x, mean_x);
    double std_y = std_dev(y, mean_y);

    // double mean_x = 0;
    // double mean_y = 0;
    // double std_x = 1;
    // double std_y = 1;

    long int ref_tstamp = tstamps[0];
    long int denumerator = 1000000000;
    int time_offset = 1;

    double params[data_length+1];
    params[0] = length;

    for(int i=1; i<length+1 ; i++){
        params[i] = (_data[i-1] - ref_tstamp) / denumerator + time_offset;
    }
    for(int i=length+1; i<length*2+1 ; i++){
        params[i] = (_data[i-1] - mean_x) / std_x;
    }
    for(int i=length*2+1; i<length*3+1 ; i++){
        params[i] = (_data[i-1] - mean_y) / std_y;
    }

    gsl_vector *theta;
    gsl_multimin_function_fdf my_func;

    my_func.n = 8;
    my_func.f = cost_f;
    my_func.df = gradient;
    my_func.fdf = grad_cost;
    my_func.params = params;

    theta = gsl_vector_alloc (8);

    for(int i = 0; i < 8; i++)
        gsl_vector_set (theta, i, 0.0);

    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;

    T = gsl_multimin_fdfminimizer_conjugate_fr;
    s = gsl_multimin_fdfminimizer_alloc (T, 8);

    gsl_multimin_fdfminimizer_set (s, &my_func, theta, 0.01, 1e-4);

    size_t iter = 0;
    int status;
    double *theta_out = new double[12];

    do
    {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate (s);

        if (status)
        break;

        status = gsl_multimin_test_gradient (s->gradient, 1e-4);

        if (status == GSL_SUCCESS){
            // printf ("Minimum found \n");
            for(int i = 0; i < 8; i++)
                theta_out[i] = gsl_vector_get (s->x, i);
        }
    //   printf ("%5d %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %10.5f\n", iter,
    //     gsl_vector_get (s->x, 0),
    //     gsl_vector_get (s->x, 1),
    //     gsl_vector_get (s->x, 2),
    //     gsl_vector_get (s->x, 3),
    //     gsl_vector_get (s->x, 4),
    //     gsl_vector_get (s->x, 5),
    //     gsl_vector_get (s->x, 6),
    //     gsl_vector_get (s->x, 7),
    //     s->f);

    }
    while (status == GSL_CONTINUE && iter < 10000);

    gsl_multimin_fdfminimizer_free (s);
    gsl_vector_free (theta);

    theta_out[8] = mean_x;
    theta_out[9] = mean_y;
    theta_out[10] = std_x;
    theta_out[11] = std_y;
    return theta_out;
}

// int main (void)
// {
//  double *par = new double[24]{1000000000,2000000000,3000000000,4000000000,5000000000,6000000000,7000000000,8000000000,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8};
//  double *rez;
//  rez = trajectory_interpolator2_cpp(par, 24, 1.01);
//  double x = (*(rez) + *(rez+1) + *(rez+2) + *(rez+3)) * *(rez+10)+ *(rez+8);
// return 0;
// }