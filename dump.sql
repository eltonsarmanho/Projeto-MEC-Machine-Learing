use teste_2;

SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS `aluno`;
DROP TABLE IF EXISTS `escola`;
DROP TABLE IF EXISTS `representante`;
DROP TABLE IF EXISTS `representante_escola`;
DROP TABLE IF EXISTS `questionario_escola`;
DROP TABLE IF EXISTS `turma`;
DROP TABLE IF EXISTS `usuario`;
DROP TABLE IF EXISTS `perfil`;
DROP TABLE IF EXISTS `perfil_usuario`;
DROP TABLE IF EXISTS `questionario_aluno`;
DROP TABLE IF EXISTS `perfil_recurso`;
DROP TABLE IF EXISTS `recurso_modulo`;
DROP TABLE IF EXISTS `ciclo_questionario`;
DROP TABLE IF EXISTS `questionario_socio_demografico`;
DROP TABLE IF EXISTS `quest_demografico`;
DROP TABLE IF EXISTS `quest_escola`;
DROP TABLE IF EXISTS `quest_aluno`;
DROP TABLE IF EXISTS `hist_perfil_usuario`;
DROP TABLE IF EXISTS `modulos_plataforma`;
DROP TABLE IF EXISTS `resp_demografico`;
SET FOREIGN_KEY_CHECKS = 1;

CREATE TABLE `aluno` (
    `id_aluno` INTEGER NOT NULL AUTO_INCREMENT,
    `id_escola` INTEGER NOT NULL,
    `cod_matricula` INTEGER,
    `nome_turma` VARCHAR(80),
    `nome_modo_ensino` VARCHAR(50),
    `nome_etapa_ensino` VARCHAR(80),
    `nome_aluno` VARCHAR(100),
    `rg` INTEGER,
    `tp_sexo` VARCHAR(50),
    `dt_nascimento` DATE,
    `nome_mae` VARCHAR(100),
    PRIMARY KEY (`id_aluno`),
    UNIQUE (`id_aluno`)
);

CREATE TABLE `escola` (
    `id_escola` INTEGER NOT NULL AUTO_INCREMENT,
    `id_turma` INTEGER NOT NULL,
    `nome_escola` VARCHAR(150) NOT NULL,
    `estado` VARCHAR(80) NOT NULL,
    `cidade` VARCHAR(100) NOT NULL,
    `endereco` VARCHAR(300) NOT NULL,
    PRIMARY KEY (`id_escola`)
);

CREATE TABLE `representante` (
    `id_representante` INTEGER NOT NULL AUTO_INCREMENT,
    `id_escola` INTEGER NOT NULL,
    `nome` VARCHAR(150) NOT NULL,
    `cpf` VARCHAR(15) NOT NULL,
    PRIMARY KEY (`id_representante`, `id_escola`)
);

CREATE TABLE `representante_escola` (
    `id` INTEGER NOT NULL AUTO_INCREMENT,
    `id_representante` INTEGER NOT NULL,
    `id_escola` INTEGER NOT NULL,
    PRIMARY KEY (`id`, `id_representante`, `id_escola`)
);

CREATE TABLE `questionario_escola` (
    `id_questionario` INTEGER NOT NULL AUTO_INCREMENT,
    `id_ciclo` INTEGER NOT NULL,
    `id_aluno` INTEGER NOT NULL,
    `dt_cadastro` DATE NOT NULL,
    `url_arquivo` TEXT NOT NULL,
    `questao1` INTEGER NOT NULL,
    `questao2` INTEGER NOT NULL,
    `questao3` INTEGER NOT NULL,
    `questao4` INTEGER NOT NULL,
    `questao5` INTEGER NOT NULL,
    `questao6` INTEGER NOT NULL,
    `questao7` INTEGER NOT NULL,
    `questao8` INTEGER NOT NULL,
    `questao9` INTEGER NOT NULL,
    `questao10` INTEGER NOT NULL,
    `questao11` INTEGER NOT NULL,
    `questao12` INTEGER NOT NULL,
    `questao13` INTEGER NOT NULL,
    `questao14` INTEGER NOT NULL,
    `questao15` INTEGER NOT NULL,
    `questao16` INTEGER NOT NULL,
    `questao17` INTEGER NOT NULL,
    `questao18` INTEGER NOT NULL,
    PRIMARY KEY (`id_questionario`)
);

CREATE TABLE `turma` (
    `id_turma` INTEGER NOT NULL AUTO_INCREMENT,
    `id_escola` INTEGER NOT NULL,
    `nome_turma` VARCHAR(50) NOT NULL,
    PRIMARY KEY (`id_turma`)
);

CREATE TABLE `usuario` (
    `cpf` INTEGER NOT NULL,
    `nome_usuario` VARCHAR(100),
    `senha` VARCHAR(100),
    `cod_regiao` CHAR(2),
    `status_usuario` CHAR(1),
    `ddd_celular_usuario` CHAR(2),
    `celular` CHAR(10),
    `data_ult_acesso` DATE,
    `nivel_usuario` INTEGER,
    `funcao_usuario` VARCHAR(100),
    `sexo` CHAR(1),
    `cod_orgao` INTEGER,
    `cod_unid` INTEGER,
    `chave_ativacao` BOOLEAN,
    `tentativa2` INTEGER,
    `rg_proposto` VARCHAR(1000),
    `usu_proposto` VARCHAR(1000),
    `obs` TEXT,
    `ungcod` CHAR(6),
    `data_inic` TIMESTAMP,
    `conectado` BOOLEAN,
    `pflcod` INTEGER,
    `suscod` CHAR(1),
    `nomeguerra` VARCHAR(20),
    `orgao` VARCHAR(100),
    `cod_mun` VARCHAR(7),
    `data_nascimento` DATE,
    `data_atualizacao` TIMESTAMP,
    `entid` INTEGER,
    `tpocod` CHAR(1),
    `carid` INTEGER,
    `arqidfoto` INTEGER,
    PRIMARY KEY (`cpf`)
);

CREATE TABLE `perfil` (
    `id_perfil` INTEGER NOT NULL AUTO_INCREMENT,
    `nivel` INTEGER NOT NULL,
    `descricao_perfil` VARCHAR(250) NOT NULL,
    PRIMARY KEY (`id_perfil`)
);

CREATE TABLE `perfil_usuario` (
    `id` INTEGER NOT NULL AUTO_INCREMENT,
    `id_perfil` INTEGER NOT NULL,
    `id_usuario` INTEGER NOT NULL,
    `id_escola` INTEGER NOT NULL,
    PRIMARY KEY (`id`)
);

CREATE TABLE `questionario_aluno` (
    `id_questionario` INTEGER NOT NULL AUTO_INCREMENT,
    `id_aluno` INTEGER NOT NULL,
    `id_ciclo` INTEGER NOT NULL,
    `dt_cadastro` DATE NOT NULL,
    `url_arquivo` TEXT NOT NULL,
    `questao1` INTEGER NOT NULL,
    `questao2` INTEGER NOT NULL,
    `questao3` INTEGER NOT NULL,
    `questao4` INTEGER NOT NULL,
    `questao5` INTEGER NOT NULL,
    `questao6` INTEGER NOT NULL,
    `questao7` INTEGER NOT NULL,
    `questao8` INTEGER NOT NULL,
    `questao9` INTEGER NOT NULL,
    `questao10` INTEGER NOT NULL,
    `questao11` INTEGER NOT NULL,
    `questao12` INTEGER NOT NULL,
    `questao13` INTEGER NOT NULL,
    `questao14` INTEGER NOT NULL,
    `questao15` INTEGER NOT NULL,
    `questao16` INTEGER NOT NULL,
    `questao17` INTEGER NOT NULL,
    `questao18` INTEGER NOT NULL,
    `questao19` INTEGER NOT NULL,
    `questao20` INTEGER NOT NULL,
    `questao21` INTEGER NOT NULL,
    `questao22` INTEGER NOT NULL,
    `questao23` INTEGER NOT NULL,
    `questao24` INTEGER NOT NULL,
    `questao25` INTEGER NOT NULL,
    `questao26` INTEGER NOT NULL,
    `questao27` INTEGER NOT NULL,
    `questao28` INTEGER NOT NULL,
    `questao29` INTEGER NOT NULL,
    `questao30` INTEGER NOT NULL,
    `questao31` INTEGER NOT NULL,
    `questao32` INTEGER NOT NULL,
    PRIMARY KEY (`id_questionario`)
);

CREATE TABLE `perfil_recurso` (
    `id` INTEGER NOT NULL AUTO_INCREMENT,
    `id_perfil` INTEGER NOT NULL,
    `id_recurso` INTEGER NOT NULL,
    PRIMARY KEY (`id`)
);

CREATE TABLE `recurso_modulo` (
    `id_recurso` INTEGER NOT NULL AUTO_INCREMENT,
    `id_modulo` INTEGER NOT NULL,
    `desc_recurso` VARCHAR(100) NOT NULL,
    PRIMARY KEY (`id_recurso`)
);

CREATE TABLE `ciclo_questionario` (
    `id_ciclo` INTEGER NOT NULL AUTO_INCREMENT,
    `nome_ciclo` VARCHAR(50) NOT NULL,
    `data_ini` DATE NOT NULL,
    `data_fim` DATE NOT NULL,
    PRIMARY KEY (`id_ciclo`)
);

CREATE TABLE `questionario_socio_demografico` (
    `id_questionario` INTEGER NOT NULL AUTO_INCREMENT,
    `id_ciclo` INTEGER NOT NULL,
    `id_aluno` INTEGER NOT NULL,
    `dt_cadastro` DATE NOT NULL,
    `url_arquivo` TEXT NOT NULL,
    `questao1` CHAR(1) NOT NULL,
    `questao2` CHAR(1) NOT NULL,
    `questao3` CHAR(1) NOT NULL,
    `questao4` CHAR(1) NOT NULL,
    `questao5` CHAR(1) NOT NULL,
    `questao6` CHAR(1) NOT NULL,
    `questao7` CHAR(1) NOT NULL,
    `questao8` CHAR(1) NOT NULL,
    `questao9` CHAR(1) NOT NULL,
    `questao10` CHAR(1) NOT NULL,
    PRIMARY KEY (`id_questionario`)
);

CREATE TABLE `quest_demografico` (
    `id_questao` INTEGER NOT NULL,
    `num_questao` INTEGER NOT NULL,
    `id_ciclo` INTEGER NOT NULL,
    `desc_questao` VARCHAR(100) NOT NULL,
    `alternativa` CHAR(1) NOT NULL,
    `desc_alternativa` VARCHAR(100) NOT NULL,
    PRIMARY KEY (`id_questao`)
);

CREATE TABLE `quest_escola` (
    `id_quest_esc` INTEGER NOT NULL,
    `id_ciclo` INTEGER NOT NULL,
    `num_quest_esc` INTEGER NOT NULL,
    `desc_quest_esc` VARCHAR(200) NOT NULL,
    PRIMARY KEY (`id_quest_esc`)
);

CREATE TABLE `quest_aluno` (
    `id_quest_aluno` INTEGER NOT NULL AUTO_INCREMENT,
    `id_ciclo` INTEGER NOT NULL,
    `num_quest_aluno` INTEGER NOT NULL,
    `desc_quest_aluno` VARCHAR(200) NOT NULL,
    PRIMARY KEY (`id_quest_aluno`)
);

CREATE TABLE `hist_perfil_usuario` (
    `id_perfil` INTEGER NOT NULL,
    `id_usuario` INTEGER NOT NULL,
    `id_escola` INTEGER NOT NULL,
    `dt_inicio` TIMESTAMP NOT NULL,
    `dt_fim` TIMESTAMP NOT NULL
);

CREATE TABLE `modulos_plataforma` (
    `id_modulo` INTEGER NOT NULL AUTO_INCREMENT,
    `nome_modulo` VARCHAR(100) NOT NULL,
    PRIMARY KEY (`id_modulo`)
);

CREATE TABLE `resp_demografico` (
    `id_questao` INTEGER NOT NULL,
    `id_ciclo` INTEGER NOT NULL,
    `alternativa` CHAR(1) NOT NULL,
    `desc_alternativa` VARCHAR(100) NOT NULL,
    PRIMARY KEY (`id_questao`)
);

ALTER TABLE `aluno` ADD CONSTRAINT teste FOREIGN KEY (`id_escola`) REFERENCES `escola`(`id_escola`);
ALTER TABLE `escola` ADD FOREIGN KEY (`id_turma`) REFERENCES `turma`(`id_turma`);
ALTER TABLE `questionario_escola` ADD FOREIGN KEY (`id_aluno`) REFERENCES `aluno`(`id_aluno`);
ALTER TABLE `questionario_escola` ADD FOREIGN KEY (`id_ciclo`) REFERENCES `ciclo_questionario`(`id_ciclo`);
ALTER TABLE `turma` ADD FOREIGN KEY (`id_escola`) REFERENCES `escola`(`id_escola`);
ALTER TABLE `perfil_usuario` ADD FOREIGN KEY (`id_escola`) REFERENCES `escola`(`id_escola`);
ALTER TABLE `perfil_usuario` ADD FOREIGN KEY (`id_usuario`) REFERENCES `usuario`(`cpf`);
ALTER TABLE `perfil_usuario` ADD FOREIGN KEY (`id_perfil`) REFERENCES `perfil`(`id_perfil`);
ALTER TABLE `questionario_aluno` ADD FOREIGN KEY (`id_ciclo`) REFERENCES `ciclo_questionario`(`id_ciclo`);
ALTER TABLE `questionario_aluno` ADD FOREIGN KEY (`id_aluno`) REFERENCES `aluno`(`id_aluno`);
ALTER TABLE `perfil_recurso` ADD FOREIGN KEY (`id_perfil`) REFERENCES `perfil`(`id_perfil`);
ALTER TABLE `perfil_recurso` ADD FOREIGN KEY (`id_recurso`) REFERENCES `recurso_modulo`(`id_recurso`);
ALTER TABLE `recurso_modulo` ADD FOREIGN KEY (`id_modulo`) REFERENCES `modulos_plataforma`(`id_modulo`);
ALTER TABLE `questionario_socio_demografico` ADD FOREIGN KEY (`id_aluno`) REFERENCES `aluno`(`id_aluno`);
ALTER TABLE `questionario_socio_demografico` ADD FOREIGN KEY (`id_ciclo`) REFERENCES `ciclo_questionario`(`id_ciclo`);