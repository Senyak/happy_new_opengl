#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <SOIL/SOIL.h> 
#include <SFML/OpenGL.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <list>

#include "model.h"

enum flags_axis {
	OX,
	OY,
	OZ,
	NUL
};

flags_axis ax = NUL;

enum flags_dir_or_pos {
	d,
	p,
};

flags_dir_or_pos repl = p;

enum flags_type_sh {
	fong,
	toon,
	rim
};

flags_type_sh sh = fong;


enum flags_type_light {
	dir,
	point,
	spot
};

flags_type_light type_light = dir;

glm::mat4 projection = glm::perspective(glm::radians(45.0f), 900.0f / 900.0f, 0.1f, 100.0f);

//камера
glm::vec3 cameraPos = glm::vec3(0.1f, 38.2f, 55.7f);
glm::vec3 cameraFront = glm::vec3(0.0f, -0.5f, -0.8f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);


//вращение
bool firstMouse = true;
float yaw = -90.0f;
float pitch = 0.0f;
float last_x = 450.0f;
float last_y = 450.0f;

//VBO
GLuint VBO_plane;

GLuint VAO_plane;

//текстуры
GLuint texture_snow;

// ID шейдерной программы
GLuint ProgramFong;

//позиция свта
glm::vec3 lightPos(0.0f, 2.0f, 10.0f);
glm::vec3 lightDirection(-36.5f, 40.5f, 62.5f);
glm::vec3 lightness(1.0f, 1.0f, 1.0f);
float conus = 12.5f;

//позиция дирижабля
glm::vec3 airship_position(0.0f, 10.0f, 0.0f);
glm::vec3 airbaloonPos(-15.0f, 15.0f, 10.0f);
float updown = 0.005f;
vector<glm::vec3> position_of_gifts;
int numb_clouds = 6;
vector<glm::vec3> position_of_clouds;
vector<float> scale_of_clouds;
bool spot_turn = true;


//fong
const char* VertexShaderFong = R"(
    #version 330 core

    layout (location = 0) in vec3 coord_pos;
    layout (location = 1) in vec3 normal_in;
    layout (location = 2) in vec2 tex_coord_in;

    out vec2 coord_tex;
	out vec3 normal;
	out vec3 frag_pos;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    void main() 
    { 
        gl_Position = projection * view * model * vec4(coord_pos, 1.0);
        coord_tex = tex_coord_in;
		frag_pos = vec3(model * vec4(coord_pos, 1.0));
		normal =  mat3(transpose(inverse(model))) * normal_in;
        //coord_tex = vec2(tex_coord_in.x, 1.0f - tex_coord_in.y); //если текстуры ннеправильно наложились
    }
    )";


//fong
const char* FragShaderFong = R"(
    #version 330 core

	struct Light {
		vec3 position;
		vec3 direction; //dir and spot
		vec3 direction_spot; //dir and spot
  
		vec3 ambient;
		vec3 diffuse;
		vec3 specular;

	//point
		float constant;
		float linear;
		float quadratic;

	//spot
		float cutOff;
		float outerCutOff;
	};

	uniform Light light;  

    in vec2 coord_tex;
    in vec3 frag_pos;
    in vec3 normal;

	out vec4 frag_color;

    uniform sampler2D texture_diffuse1;
	uniform vec3 viewPos;
	uniform vec3 airshipPos;
	uniform int shininess;
	uniform int spot_turn;

    void main()  
    {
		vec3 norm = normalize(normal);
		vec3 lightDir;

		lightDir = normalize(-light.direction);  //dir
		
		vec3 ambient = light.ambient * texture(texture_diffuse1, coord_tex).rgb; 

		float diff = max(dot(norm, lightDir), 0.0);
		vec3 diffuse = light.diffuse * (diff * texture(texture_diffuse1, coord_tex).rgb); 

		vec3 viewDir = normalize(viewPos - frag_pos);
		vec3 reflectDir = reflect(-lightDir, norm);  

		float spec = pow(max(dot(viewDir, reflectDir), 0.0), 1);
		vec3 specular = light.specular * (spec * texture(texture_diffuse1, coord_tex).rgb); 

		vec3 result = (ambient*2 + diffuse + specular);
		result += ambient;

		if(spot_turn == 1)
		{
			lightDir = normalize(airshipPos - frag_pos);  //point and spot

			diff = max(dot(norm, lightDir), 0.0);
			diffuse = vec3(0.7f, 0.6f, 0.2f) * (diff * texture(texture_diffuse1, coord_tex).rgb); 

		    reflectDir = reflect(-lightDir, norm);  
			spec = pow(max(dot(viewDir, reflectDir), 0.0), 1);
			specular = vec3(0.8f, 0.7f, 0.2f) * (spec * texture(texture_diffuse1, coord_tex).rgb);

			float distance    = length(airshipPos - frag_pos);
			float attenuation = 1.0 / (light.constant + light.linear * distance 
								+ light.quadratic * (distance * distance));
				
			diffuse  *= attenuation;
			specular *= attenuation;  
			
			float theta = dot(lightDir, normalize(-light.direction_spot)); 
			float epsilon   = light.cutOff - light.outerCutOff;
			float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);

			diffuse  *= intensity;
			specular *= intensity;

			result +=  diffuse + specular;
			result +=  diffuse;
			
		}

		frag_color = vec4(result, 1.0);
    } 
)";


void checkOpenGLerror()
{
	GLenum err = glGetError();
	if (err != GL_NO_ERROR)
	{
		std::cout << "OpenGL error " << err << std::endl;
	}
}

void ShaderLog(unsigned int shader)
{
	int infologLen = 0;
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLen);
	GLint vertex_compiled;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &vertex_compiled);

	if (infologLen > 1)
	{
		int charsWritten = 0;
		std::vector<char> infoLog(infologLen);
		glGetShaderInfoLog(shader, infologLen, &charsWritten, infoLog.data());
		std::cout << "InfoLog: " << infoLog.data() << std::endl;
	}

	if (vertex_compiled != GL_TRUE)
	{
		GLsizei log_length = 0;
		GLchar message[1024];
		glGetShaderInfoLog(shader, 1024, &log_length, message);
		std::cout << "InfoLog2: " << message << std::endl;
	}

}

void InitShader()
{
	GLuint vShaderFong = glCreateShader(GL_VERTEX_SHADER); 
	// Передаем исходный код
	glShaderSource(vShaderFong, 1, &VertexShaderFong, NULL); 
	// Компилируем шейдер
	glCompileShader(vShaderFong);
	std::cout << "vertex shader f\n";
	// Функция печати лога шейдера
	ShaderLog(vShaderFong); 

	//-----------------------
	
	// Создаем фрагментный шейдер
	GLuint fShaderFong = glCreateShader(GL_FRAGMENT_SHADER);
	// Передаем исходный код
	glShaderSource(fShaderFong, 1, &FragShaderFong, NULL);
	// Компилируем шейдер
	glCompileShader(fShaderFong);
	std::cout << "fragment shader f\n";
	// Функция печати лога шейдера
	ShaderLog(fShaderFong);

	ProgramFong = glCreateProgram();
	glAttachShader(ProgramFong, vShaderFong);
	glAttachShader(ProgramFong, fShaderFong);

	// Линкуем шейдерную программу
	glLinkProgram(ProgramFong);

	// Проверяем статус сборки
	int link_ok;
	glGetProgramiv(ProgramFong, GL_LINK_STATUS, &link_ok);

	if (!link_ok)
	{
		std::cout << "error attach shaders \n";
		return;
	}

	checkOpenGLerror();
}



void Init()
{
	// Шейдеры
	InitShader();

	//включаем тест глубины
	glEnable(GL_DEPTH_TEST);
}


void Draw(sf::Clock clock, vector<Model> modelka, GLint shader, int count)
{
	glm::mat4 view = glm::mat4(1.0f);
	glm::mat4 model = glm::mat4(1.0f);

	glUseProgram(shader); // Устанавливаем шейдерную программу текущей

	view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

	projection = glm::perspective(glm::radians(45.0f), 900.0f / 900.0f, 0.1f, 1000.0f);
	
	glUniform3f(glGetUniformLocation(shader, "airshipPos"), airship_position.x, airship_position.y, airship_position.z);
	glUniform3f(glGetUniformLocation(shader, "light.position"), lightPos.x, lightPos.y, lightPos.z);
	
	glUniform3f(glGetUniformLocation(shader, "viewPos"), cameraPos.x, cameraPos.y, cameraPos.z);
	
	glUniform3f(glGetUniformLocation(shader, "light.ambient"), 0.2f, 0.2f, 0.2f);
	glUniform3f(glGetUniformLocation(shader, "light.diffuse"), 0.9f, 0.9f, 0.9);
	glUniform3f(glGetUniformLocation(shader, "light.specular"), 1.0f, 1.0f, 1.0f);
	glUniform1i(glGetUniformLocation(shader, "spot_turn"), spot_turn);
	glUniform1i(glGetUniformLocation(shader, "shininess"), 16);
	
	glUniform3f(glGetUniformLocation(shader, "light.direction"), lightDirection.x, lightDirection.y, lightDirection.z);
	glUniform3f(glGetUniformLocation(shader, "light.direction_spot"), 0.0f, -28.0f, -43.0f);

	glUniform1f(glGetUniformLocation(shader, "light.constant"), 1.0f);
	glUniform1f(glGetUniformLocation(shader, "light.linear"), 0.045f);
	glUniform1f(glGetUniformLocation(shader, "light.quadratic"), 0.0075f);

	glUniform1f(glGetUniformLocation(shader, "light.cutOff"), glm::cos(glm::radians(conus)));
	glUniform1f(glGetUniformLocation(shader, "light.outerCutOff"), glm::cos(glm::radians(conus * 1.4f)));

	glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

	glUniform1i(glGetUniformLocation(shader, "shininess"), 2);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_snow);
	glUniform1i(glGetUniformLocation(shader, "texture_diffuse1"), 0);
	glBindVertexArray(VAO_plane);

	model = glm::scale(model, glm::vec3(1.5f, 1.0f, 1.0f));

	glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm::value_ptr(model));

	glDrawArrays(GL_TRIANGLES, 0, 36);

	//дирижабль
	model = glm::mat4(1.0f);
	model = glm::translate(model, glm::vec3(airship_position.x, airship_position.y, airship_position.z));
	model = glm::rotate(model, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::scale(model, glm::vec3(1.5f, 1.5f, 1.5f)); 
	model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));

	glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm::value_ptr(model));

	modelka[0].Draw(shader, count);

	//подарки
	for (int i = 0; i < position_of_gifts.size(); i++)
	{
		if (position_of_gifts[i].y > 0.0f)
		{
			model = glm::mat4(1.0f);
			model = glm::translate(model, position_of_gifts[i]);
			model = glm::scale(model, glm::vec3(0.08f, 0.08f, 0.08f));
			model = glm::rotate(model, clock.getElapsedTime().asSeconds() * glm::radians(60.0f), glm::vec3(1.0f, 0.0f, 1.0f));
			position_of_gifts[i] -= glm::vec3(0.0f, 0.03f, 0.0f);

			glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm::value_ptr(model));
			modelka[1].Draw(shader, count);
		}

	}

	if (position_of_gifts.size() > 0 && position_of_gifts[0].y < 0.0f)
	{
		position_of_gifts.erase(position_of_gifts.begin());
	}

	//облака
	for (int i = 0; i < numb_clouds; i++)
	{
		model = glm::mat4(1.0f);
		model = glm::translate(model, position_of_clouds[i]);
		model = glm::scale(model, glm::vec3(18.0f + scale_of_clouds[i], 18.0f + scale_of_clouds[i], 18.0f + scale_of_clouds[i]));
		model = glm::rotate(model, clock.getElapsedTime().asSeconds() * glm::radians(scale_of_clouds[i] * 5.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		
		glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm::value_ptr(model));
		modelka[2].Draw(shader, count);
	}

	//воздушный шар
	airbaloonPos += glm::vec3(0.0f, updown, 0.0f);

	if (airbaloonPos.y > 18.0f || airbaloonPos.y < 15.0f)
	{
		updown *= -1.0f;
	}

	model = glm::mat4(1.0f);
	model = glm::translate(model, airbaloonPos);
	model = glm::scale(model, glm::vec3(0.8f, 0.8f, 0.8f));

	glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm::value_ptr(model));

	modelka[3].Draw(shader, count);

	//елка
	model = glm::mat4(1.0f);
	model = glm::translate(model, glm::vec3(15.0f, 0.0f, -3.0f));
	model = glm::scale(model, glm::vec3(0.005f, 0.005f, 0.005f));
	model = glm::rotate(model, clock.getElapsedTime().asSeconds() * glm::radians(20.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm::value_ptr(model));

	modelka[4].Draw(shader, count);

	//пингвин
	model = glm::mat4(1.0f);
	model = glm::translate(model, glm::vec3(-15.0f, 4.0f, 10.0f));
	model = glm::scale(model, glm::vec3(0.015f, 0.015f, 0.015f));
	model = glm::rotate(model,glm::radians(30.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm::value_ptr(model));

	modelka[5].Draw(shader, count);

	glUseProgram(0); // Отключаем шейдерную программу


	checkOpenGLerror();
}


// Освобождение буфера
void ReleaseVBO()
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Освобождение шейдеров
void ReleaseShader()
{
	// Передавая ноль, мы отключаем шейдерную программу
	glUseProgram(0);
	// Удаляем шейдерные программы
	glDeleteProgram(ProgramFong);
}


void Release()
{
	// Шейдеры
	ReleaseShader();
	// Вершинный буфер
	ReleaseVBO();
}


int main()
{
	std::setlocale(LC_ALL, "Russian");

	sf::Window window(sf::VideoMode(900, 900), "Fill", sf::Style::Default, sf::ContextSettings(24));
	window.setVerticalSyncEnabled(true);
	window.setActive(true);
	
	glewInit();
	glGetError(); // сброс флага GL_INVALID_ENUM

	sf::Clock clock;

	vector<Model> models;
	
	Model flyship("flyship/FlyShip1_01.obj");
	models.push_back(flyship);

	Model box("box1/b1.obj");
	models.push_back(box); 

	Model cloud("cloud2/cloud.obj");
	models.push_back(cloud);

	Model airbaloon("airbaloon/airbaloon.obj");
	models.push_back(airbaloon);

	Model tree("chrtree/tree.obj");
	models.push_back(tree);

	Model penguin("penguin/penguin.obj");
	models.push_back(penguin);



	srand(time(0));

	for (int i = 0; i < numb_clouds; i++)
	{

		int start = -20;
		int end = 20;
		float x = (rand() % (end - start + 1) + start) * 1.0f;

		start = -25;
		end = 15;
		float z = (rand() % (end - start + 1) + start) * 1.0f;

		start = 24;
		end = 25;
		float y = (rand() % (end - start + 1) + start) * 1.0f;

		position_of_clouds.push_back(glm::vec3(x, y, z));

		start = -5;
		end = 5;
		y = (rand() % (end - start + 1) + start) * 1.0f;

		scale_of_clouds.push_back(y);
	}


	Init();
	float plane[] = {
		 25.0f, -0.5f,  25.0f,  0.0f, 1.0f, 0.0f,  5.0f,  0.0f,
		-25.0f, -0.5f,  25.0f,  0.0f, 1.0f, 0.0f,   0.0f,  0.0f,
		-25.0f, -0.5f, -25.0f,  0.0f, 1.0f, 0.0f,   0.0f, 5.0f,

		 25.0f, -0.5f,  25.0f,  0.0f, 1.0f, 0.0f,  5.0f,  0.0f,
		-25.0f, -0.5f, -25.0f,  0.0f, 1.0f, 0.0f,   0.0f, 5.0f,
		 25.0f, -0.5f, -25.0f,  0.0f, 1.0f, 0.0f,  5.0f, 5.0f
	};

	//объявляем массив атрибутов и буфер

	glGenVertexArrays(1, &VAO_plane);
	glGenBuffers(1, &VBO_plane);
	glBindVertexArray(VAO_plane);
	// передаем вершины в буфер
	glBindBuffer(GL_ARRAY_BUFFER, VBO_plane);
	glBufferData(GL_ARRAY_BUFFER, sizeof(plane), plane, GL_STATIC_DRAW);

	// Подключаем массив аттрибутов с указанием на каких местах кто находится
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));

	glBindVertexArray(0);

	// создаем текстуру
	glGenTextures(1, &texture_snow);

	// связываем с типом текступы
	glBindTexture(GL_TEXTURE_2D, texture_snow);

	// настроки отображения текстуры при выходе за пределы диапазона текстуры
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	// настройки отображения текстуры в зависимости от удаления или приближения обьекта
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// грузим картинку
	int width, height;
	unsigned char* image = SOIL_load_image("snow.jpg", &width, &height, 0, SOIL_LOAD_RGB);

	//создаем текстуру
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
	SOIL_free_image_data(image);

	//отключаем привязку к текстуре
	glBindTexture(GL_TEXTURE_2D, 0);


	while (window.isOpen())
	{
		sf::Event event;

		while (window.pollEvent(event))
		{

			float camera_speed = 0.5f;

			if (event.type == sf::Event::Closed)
				window.close();
			if (event.type == sf::Event::KeyPressed)
			{
				if (event.key.code == sf::Keyboard::Escape)
				{
					window.close();
					break;
				}

				if (event.key.code == sf::Keyboard::W)
				{
					airship_position -= glm::vec3(0.0f, 0.0f, 0.5f);
				}
				if (event.key.code == sf::Keyboard::S)
				{
					airship_position += glm::vec3(0.0f, 0.0f, 0.5f);
				}
				if (event.key.code == sf::Keyboard::A)
				{
					airship_position -= glm::vec3(0.5f, 0.0f, 0.0f);
				}
				if (event.key.code == sf::Keyboard::D)
				{
					airship_position += glm::vec3(0.5f, 0.0f, 0.0f);
				}
				if (event.key.code == sf::Keyboard::Q)
				{
					airship_position += glm::vec3(0.0f, 0.5f, 0.0f);
				}
				if (event.key.code == sf::Keyboard::E)
				{
					airship_position -= glm::vec3(0.0f, 0.5f, 0.0f);
				}

				if (event.key.code == sf::Keyboard::F)
				{
					spot_turn = !spot_turn;

				}

				if (event.key.code == sf::Keyboard::K)
				{
					conus += 0.5f;
					std::cout << conus << std::endl;
				}
				if (event.key.code == sf::Keyboard::L)
				{
					conus -= 0.5f;
					std::cout << conus << std::endl;
				}

				if (event.key.code == sf::Keyboard::Y)
				{
					cameraPos -= camera_speed * glm::vec3(0.0f, 0.0f, 1.0f);
				}
				if (event.key.code == sf::Keyboard::H)
				{
					cameraPos += camera_speed * glm::vec3(0.0f, 0.0f, 1.0f);
				}
				if (event.key.code == sf::Keyboard::G)
				{
					cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * camera_speed;
				}
				if (event.key.code == sf::Keyboard::J)
				{
					cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * camera_speed;
				}
				if (event.key.code == sf::Keyboard::T)
				{
					cameraPos += camera_speed * cameraUp;
				}
				if (event.key.code == sf::Keyboard::U)
				{
					cameraPos -= camera_speed * cameraUp;
				}

			}
			if (event.type == sf::Event::MouseButtonPressed)
			{

				if (event.mouseButton.button == sf::Mouse::Left)
				{
					position_of_gifts.push_back(airship_position);
				}
			}
			else if (event.type == sf::Event::Resized)
				glViewport(0, 0, event.size.width, event.size.height);
		}

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.1f, 0.05f, 0.25f, 1.0f);
		
		Draw(clock, models, ProgramFong, 1);

		window.display();
	}

	Release();
	return 0;
}