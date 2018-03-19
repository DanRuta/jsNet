#include <vector>

template<class T>
void printv(T values) {
    EM_ASM(window.printfVector = []);

    for (int i=0; i<values.size(); i++) {
        EM_ASM_({
            window.printfVector.push($0)
        }, values[i]);
    }

    EM_ASM(console.log(window.printfVector));
}

template<class T>
void printv(std::vector<std::vector<T>> values) {
    EM_ASM(window.printfVector = []);

    for (int i=0; i<values.size(); i++) {

        EM_ASM_({window.printfVector[$0] = []}, i);

        for (int j=0; j<values[i].size(); j++) {
            EM_ASM_({
                window.printfVector[$0].push($1)
            }, i, values[i][j]);
        }
    }

    EM_ASM(console.log(window.printfVector));
}


template<class T>
void printv(std::vector<std::vector<std::vector<T>>> values) {
    EM_ASM(window.printfVector = []);

    for (int i=0; i<values.size(); i++) {

        EM_ASM_({window.printfVector[$0] = []}, i);

        for (int j=0; j<values[i].size(); j++) {

            EM_ASM_({window.printfVector[$0][$1] = []}, i, j);

            for (int k=0; k<values[i][j].size(); k++) {
                EM_ASM_({
                    window.printfVector[$0][$1].push($2)
                }, i, j, values[i][j][k]);
            }
        }
    }

    EM_ASM(console.log(window.printfVector));
}


template<class T>
void printv(T values[], int size) {
    EM_ASM(window.printfVector = []);

    for (int i=0; i<size; i++) {
        EM_ASM_({
            window.printfVector.push($0)
        }, values[i]);
    }

    EM_ASM(console.log(window.printfVector));
}