#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <iostream>
#include "semafor.h"

using namespace std;

class Peron;

class Pociag {
private:
    string m_nazwa;
    Peron &peron;
    Peron &peronZapasowy;

public:
    Pociag(string n, Peron &p1, Peron &p2);
    void operator()();
    string nazwa();
};

class Peron {
private:
    string nazwa;
    bool pusty;

public:
    Semafor &s;

    Peron(string n, Semafor &s) : nazwa(n), s(s), pusty(true) {}

    void przyjazd(Pociag& p) {

        if (pusty) {
            string info;
            info.append("Pociag ");
            info.append(p.nazwa());
            info.append(" wjezdza na peron ");
            info.append(nazwa);
            info.append("\n");
            cout << info;
            } else {
                string info;
                info.append("Pociag ");
                info.append(p.nazwa());
                info.append(" spowodowal katastrofe na peronie ");
                info.append(nazwa);
                info.append("\n");
                cout << info;
            }
        pusty = false;
    }

    void odjazd(Pociag& p) {
        string info;
        info.append("Pociag ");
        info.append(p.nazwa());
        info.append(" opuszcza peron ");
        info.append(nazwa);
        info.append("\n");
        cout << info;
        pusty = true;
        s.sygnalizuj();
    }

    bool czyPusty()
    {
        return pusty;
    }
};

minstd_rand generator;
uniform_int_distribution<int> zakres(0, 500);

Pociag::Pociag(string n, Peron &p1, Peron &p2): m_nazwa(n), peron(p1), peronZapasowy(p2) {}

string Pociag::nazwa() {return m_nazwa;}

void Pociag::operator()() {
    for (int i = 0; i < 3; i++) {
        this_thread::sleep_for(chrono::milliseconds(zakres(generator)));
        //pociag probuje wjechac na peron preferowany
        if (peron.s.probuj_czekac()) {
            //peron jest wolny wiec pociag wjezdza
            peron.przyjazd(*this);
            //pasazerowie wsiadaja/wysiadaja
            this_thread::sleep_for(chrono::milliseconds(zakres(generator)));
            //pociag odjezdza
            peron.odjazd(*this);
        } //jezeli sie nie udalo to wjezdza na peron alternatywny
        else {
            peronZapasowy.s.czekaj();
            //peron jest wolny wiec pociag wiejdza
            peronZapasowy.przyjazd(*this);
            //pasazerowie wsiadaja/wysiadaja
            this_thread::sleep_for(chrono::milliseconds(zakres(generator)));
            //pociag odjezdza
            peronZapasowy.odjazd(*this);
        }
    }
}

int main(int argc, char* argv[]) {
//na stacji sa 3 perony
    Semafor s1, s2, s3;
    Peron p1("p1", s1), p2("p2", s2), p3("p3", s3);

    Pociag pociag1("pociag1", p2, p1);
    Pociag pociag2("pociag2", p2, p3);
    Pociag pociag3("pociag3", p2, p1);
    Pociag pociag4("pociag4", p2, p3);

    thread w1(pociag1);
    thread w2(pociag2);
    thread w3(pociag3);
    thread w4(pociag4);

    w1.join();
    w2.join();
    w3.join();
    w4.join();

return (EXIT_SUCCESS);
}
