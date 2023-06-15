%%
%Opgave 2b

%load data
data = load('SSTA_1901_2022.txt');

data(data==0) = NaN;

lon = data(:,1);
lat = data(:,2);


%plot lort
h = figure(1);
lat_min = 56;
lat_max = 84;
lon_min = -90;
lon_max = -0;
m_proj(['Transverse Mercator'],'long',[lon_min lon_max],'lat',[lat_min lat_max],'rectbox','off'); 

axis tight manual
filename = 'opgave1trans2test.gif';

x0=650;
y0=150;
width=600;
height=350;
set(gcf,'position',[x0,y0,width,height])

for n = 1:122
    m_scatter(lon,lat,50,data(:,n+2),'sq','fill')
m_gshhs(1, 'color','k');
m_grid('xtick',6,'ytick',4,'linewi',1,'tickstyle','dm','tickdir','in','XAxisLocation','bottom','yaxisloc','left');
title(['year: ' num2str(1900+n*1)], 'FontSize',15) % Titel paa plottet
xlabel('Longitude in degrees')
ylabel('Latitude in degrees')
ax=gca;
colormap(jet(256))
colorbar
set(get(colorbar,'label'),'string','SSTA in Degrees','Rotation',90.0);
caxis([-2 2])

      % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if n == 1 
          imwrite(imind,cm,filename,'gif','DelayTime',0.3,'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','DelayTime',0.3,'WriteMode','append'); 
      end 
end

%%
%Opgave 1.c sjov
%load data
data = load('SSTA_1901_2022.txt');

data(data==0) = NaN;

%gennemsnit pr. år for 2012 - 2022
for n1 = 1:11
    a(n1)=mean(data(:,113+n1*1),'omitnan');
    
end

%gennemsnit pr. år for 1961-1990 (referenceperioden)
for n2 = 1:92
    b(n2)=mean(data(:,2+n2*1),'omitnan');
    
end

%gennemsnit for perioden 2012 - 2022
mean(a)
%gennemsnit for perioden 1961 - 1990
mean(b)

%%
%gennemsnit pr. år for 1918 - 1922
for n3 = 1:5
    c(n3)=mean(data(:,19+n3*1),'omitnan');
    
end

%gennemsnit pr. år for hele datasættet
for n4 = 1:122
    d(n4)=mean(data(:,2+n4*1),'omitnan');
    
end

%gennemsnit for perioden 1918 - 1922
mean(c)
%gennemsnit for perioden 1901 - 2022
mean(d)

%%
%gennemsnit for perioden 1923 - 1928 
for n5 = 28:32
    e(n5)=mean(data(:,2+n5*1),'omitnan');
    
end

%gennemsnit for perioden 19123 - 1938
mean(e)