#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <iomanip>

using namespace std;

class Animation{
	public:
			Animation(int speed, const string & init){
				
				new_speed = speed;
				new_init = init;
				init_size = init.length();
				particle_num = 0;

				for(int i = 0; i < init_size; i++){
					if(init[i] == 'R'){
						status.push_back(1); // store direction of particle
						status.push_back(i);
						particle.push_back(status);
					}
					else if (init[i] == 'L'){
						status.push_back(0);
					  status.push_back(i);
				   	particle.push_back(status);
					}
				}
			}
			
			vector<string> animate(int speed, const string& init){
			int finish_count = 0;
			vector<string> str_vect;
			
			while(finish_count < particle.size()){
				
				// Every iteration recalculates the current position of each particle
				for(int i = 0; i < particle.size(); i++){
						if(particle.at(i).at(0) == 1){
							particle.at(i).at(1) += speed;
							if (particle.at(i).at(1) > init.length()){
								finish_count++;
							}
						}
						else if(particle.at(i).at(0) == 0){
							particle.at(i).at(1) -= speed;
							if (particle.at(i).at(1) < 0){
							  finish_count++;
							}
						}
				}
				}
				
					string print_string(init_size,'.');

					for(int i = 0; i < particle.size(); i++){
						if(particle.at(i).at(1) > 0 && particle.at(i).at(1) < init.length()){
							cout << "Particle at " << particle.at(i).at(1) << endl;
							print_string = print_string.replace(particle.at(i).at(1),1,"X");
						}
					}
					// Pushes every time snapshot into the string vector
					str_vect.push_back(print_string);
					return str_vect;
			}

			// Print output vector
			void print_animation(vector<string> str_vect){

				cout << "{";
				for(int i = 0; i < str_vect.size(); i++){

				cout << "\"" << str_vect[i] << "\",";
				}
				cout << "}" << endl;
			}

	private:
			int new_speed;
			int init_size;
			string new_init;
			int particle_num;
			vector<int>  status; // Status query: status 1 = position, status 2 = direction
			vector<vector<int> > particle;
};

int main(int argc, char* argv[]){

	stringstream ss;
	ss.str ("2,  \"..R....\"");
	string init;
	string pos;
	int speed;
	vector<string> output;
	
	getline( ss, init, ',');
	getline( ss, pos, ',');
	remove(pos.begin(), pos.end(), ' ');
	pos.erase(remove(pos.begin(), pos.end(), '\"'),pos.end());
	pos = pos.substr(0,pos.size()-1);
	cout << "Extracted " << init << endl;
	cout << "Next extract is " << pos << endl;

	speed = atoi(init.c_str());

	cout << "Speed is " << speed << endl;
	cout << "Size of string is " << pos.length() << endl;
	
	Animation am1(speed, pos);
	output = am1.animate(speed, pos);
	am1.print_animation(output);

	return 0;
}
