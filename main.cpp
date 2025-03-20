#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace std;


// Read the data from assigned file.
vector<pair<int, vector<double>>> read_data(const string& filename) {

    vector<pair<int, vector<double>>> data;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {

        // Using stringstream to parse class_llabel and features
        istringstream iss(line);
        double class_label;

        // Parse the first column as class_label either 1 or 2.
        iss >> class_label;
        vector<double> features;
        double feature;

        // Parse the rest of column as features.
        while (iss >> feature) {
            features.push_back(feature);
        }

        // Store as (int class_label, vector<double> features)
        data.push_back(make_pair(static_cast<int>(class_label), features));
    }


    return data;

}

// Compute accuracy
double compute_accuracy(const vector<pair<int, vector<double>>>& data, const vector<int>& selected_features) {

    int correct = 0;
    double accuracy;

    for (int i = 0; i < data.size(); i++) {

        const auto& test_instance = data[i];
        // Initialize min_dist to a nearly unreachable large number 
        double min_dist = 1000.0;
        // Initialize the label to invalid label
        int nearest_label = -1;

        for (int j = 0; j < data.size(); j++) {

            // Make sure it doesn't compare with itself
            if (i == j) {
                continue;
            }

            // Compute Euclidean distance using selected features
            double dist = 0.0;
            for (int f : selected_features) {
                double diff = test_instance.second[f] - data[j].second[f];
                dist += diff * diff;
            }

            dist = sqrt(dist);

            if (dist < min_dist) {
                min_dist = dist;
                nearest_label = data[j].first;
            }
        }

        if (nearest_label == test_instance.first) {
            correct++;
        }
    }

    // Accuracy = correct Prediction / the total number of instances
    accuracy = static_cast<double>(correct) / data.size();

    return accuracy;
}


// Forward Selection
pair<vector<int>, double> forward_selection(const vector<pair<int, vector<double>>>& data) {

    // Initialization
    int num_features;
    vector<int> selected;
    int best_feature;
    double best_accuracy;
    // tempAccuracy = accuracy * 100 only for printing purpose
    double tempAccuracy;
    vector<int> candidate;
    double accuracy;
    vector<pair<vector<int>, double>> history;
    double max_accuracy;
    vector<int> best_subset;


    if (!data.empty()) {
        num_features = data[0].second.size();
    }

    cout << "Forward Selection starting..." << endl;


    for (int step = 0; step < num_features; step++) {
        // Reset
        best_feature = -1;
        best_accuracy = -1.0;

        for (int f = 0; f < num_features; f++) {

            // Clean up candiate
            candidate.erase(candidate.begin(), candidate.end());


            if (find(selected.begin(), selected.end(), f) != selected.end()) {
                continue;
            }

            candidate = selected;
            candidate.push_back(f);
            accuracy = compute_accuracy(data, candidate);

            cout << "Using feature(s) [";

            for (int i = 0; i < candidate.size(); i++) {
                cout << (candidate[i] + 1);

                if (i != candidate.size() - 1) {
                    cout << ",";
                }
            }

            tempAccuracy = accuracy * 100;
            cout << "] accuracy: " << tempAccuracy << "%" << endl;

            if (accuracy > best_accuracy || (accuracy == best_accuracy && f < best_feature)) {
                best_accuracy = accuracy;
                best_feature = f;
            }
        }

        if (best_feature == -1) {
            break;
        }

        selected.push_back(best_feature);
        history.push_back(make_pair(selected, best_accuracy));

        cout << "Added feature " << (best_feature + 1) << ", current feature set: [";

        for (int i = 0; i < selected.size(); i++) {

            cout << (selected[i] + 1);

            if (i != selected.size() - 1) {
                cout << ",";
            }
        }

        tempAccuracy = best_accuracy * 100;
        cout << "], accuracy: " << tempAccuracy << "%" << endl;
    }

    max_accuracy = -1.0;

    for (const auto& entry : history) {
        if (entry.second > max_accuracy || (entry.second == max_accuracy && entry.first.size() < best_subset.size())) {
            max_accuracy = entry.second;
            best_subset = entry.first;
        }
    }

    return {best_subset, max_accuracy};
}


// Backward Elimination
pair<vector<int>, double> backward_elimination(const vector<pair<int, vector<double>>>& data) {

    // Initialization
    int num_features;
    vector<pair<vector<int>, double>> history;
    // tempAccuracy = accuracy * 100 only for printing purpose
    double tempAccuracy;
    double initial_acc;
    int worst_feature;
    double best_accuracy;
    int f;
    vector<int> candidate;
    double accuracy;
    double max_accuracy;
    vector<int> best_subset;

    if (!data.empty()) {
        num_features = data[0].second.size();
    }

    // Creating selected features with a size of features
    vector<int> selected(num_features);

    for(int i = 0; i < selected.size(); i++) {
        selected[i] = i;
    }

    
    initial_acc = compute_accuracy(data, selected);

    // Have to use make_pair in order for push_back to work properly.
    history.push_back(make_pair(selected, initial_acc));


    cout << "Backward Elimination starting..." << endl;
    
    cout << "Initial feature set [";

    for (int i = 0; i < selected.size(); i++) {

        cout << (selected[i] + 1);

        if (i != selected.size() - 1) {
            cout << ",";
        }
    }

    tempAccuracy = initial_acc * 100;
    cout << "] accuracy: " << tempAccuracy << "%" << endl;



    for (int step = 0; step < num_features - 1; step++) {
        // Reset
        worst_feature = -1;
        best_accuracy = -1.0;

        for (auto it = selected.begin(); it != selected.end(); it++) {

            f = *it;
            // Clean up candiate
            candidate.erase(candidate.begin(), candidate.end());


            for (int sf : selected) {
                if (sf != f) {
                    candidate.push_back(sf);
                }
            }

            accuracy = compute_accuracy(data, candidate);

            tempAccuracy = accuracy * 100;
            cout << "Removing feature " << (f + 1) << ", accuracy: " << tempAccuracy << "%" << endl;

            if (accuracy > best_accuracy || (accuracy == best_accuracy && f > worst_feature)) {
                best_accuracy = accuracy;
                worst_feature = f;
            }
        }

        if (worst_feature == -1) {
            break;
        }

        selected.erase(remove(selected.begin(), selected.end(), worst_feature), selected.end());
        history.push_back(make_pair(selected, best_accuracy));

        cout << "Removed feature " << (worst_feature + 1) << ", current feature set: [";


        for (int i = 0; i < selected.size(); i++) {
            cout << (selected[i] + 1);
            if (i != selected.size() - 1) {
                cout << ",";
            }
        }

        tempAccuracy = best_accuracy * 100;
        cout << "], accuracy: " << tempAccuracy << "%" << endl;
    }

    max_accuracy = -1.0;
    for (const auto& entry : history) {
        if (entry.second > max_accuracy || (entry.second == max_accuracy && entry.first.size() < best_subset.size())) {
            max_accuracy = entry.second;
            best_subset = entry.first;
        }
    }

    return {best_subset, max_accuracy};

}





int main() {
    int userC;
    string fileName;
    double tempAccuracy;

    cout << "Type in the name of the file to test:" << endl;
    cin >> fileName;

    cout << "Type the number of the algorithm you want to run" << endl;
    cout << "1) Forward Selection" << endl;
    cout << "2) Backward Elimination" << endl;
    cin >> userC;

    vector<pair<int, vector<double>>> data = read_data(fileName);


    cout << "This dataset has " << data[0].second.size() << " features (not including the class attribute), with " << data.size() << " instances.\n" << endl;


    // Forward Selection
    if (userC == 1) {

        auto fs_result = forward_selection(data);

        cout << "\nForward Selection Best Result:" << endl;
        cout << "Features: [";

        // Reorder the best subset by the feature number.
        sort(fs_result.first.begin(), fs_result.first.end());

        for (int i = 0; i < fs_result.first.size(); i++) {

            cout << (fs_result.first[i] + 1);

            if (i != fs_result.first.size() - 1) {
                cout << ",";
            }
        }

        cout << "], Accuracy: " << fs_result.second * 100 << "%" << "\n" << endl;
    }


    // Backward Elimination
    else if (userC == 2) {

        auto be_result = backward_elimination(data);

        cout << "\nBackward Elimination Best Result:" << endl;
        cout << "Features: [";

        sort(be_result.first.begin(), be_result.first.end());


        for (int i = 0; i < be_result.first.size(); i++) {
            cout << (be_result.first[i] + 1);

            if (i != be_result.first.size() - 1) {
                cout << ",";
            }
        }

        cout << "], Accuracy: " << be_result.second * 100 << "%" << endl;
    }

    else {
        cout << "Invalid choice" << endl;
    }



    return 0;
}