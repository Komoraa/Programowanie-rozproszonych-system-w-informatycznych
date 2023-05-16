#ifndef SEMAFOR_H_
#define SEMAFOR_H_
#include <mutex>
#include <condition_variable>

using namespace std;

//sluzy do zarzadzania dostepem do ograniczonych zasobow

class Semafor
{
public:
    Semafor(int _count=1);

    //zwalnia zasob
    void sygnalizuj();

    //zajmuje zasob (jezeli nie ma wolnych
    //to czeka na zwolnienie zasobu)
    void czekaj();

    //probuje zajac zasob, zwraca true jezeli
    //sie udalo, false jezeli wszystkie
    //zasoby sa zajete
    bool probuj_czekac();

    //zwraca biezaca wartosc licznika
    int wartosc() const;

private:
    //licznik dostepnych zasobow
    int licznikZasobow;

    //synchronizacja watkow
    mutex mutexLicznika;
    condition_variable warunekLicznikRoznyOdZera;
};

#endif /*SEMAFOR_H_*/
